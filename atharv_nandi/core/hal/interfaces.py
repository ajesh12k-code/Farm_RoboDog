"""ros2_control-style hardware abstraction for Atharv Nandi (WPCAP; docs/hal-sim-to-real.md option B with the
names and semantics of option A).

This is a ROS-free Python mirror of the ros2_control concepts, so the sim, the bench stub and a later real
servo-bus driver share one contract and a move to the C++ ``ros2_control`` stack changes no consumer:

==========================  ===========================================================================
ros2_control                here
==========================  ===========================================================================
``<ros2_control>`` URDF tag  ``config/hal.yaml`` -> ``HardwareInfo`` / ``ComponentInfo`` / ``InterfaceInfo``
``hardware_interface::       ``SystemInterface``: ``on_init`` / ``on_configure`` / ``on_activate`` /
SystemInterface``            ``on_deactivate`` / ``on_cleanup`` / ``on_shutdown`` / ``on_error``,
                             ``export_state_interfaces`` / ``export_command_interfaces``, ``read`` / ``write``,
                             ``prepare_command_mode_switch`` / ``perform_command_mode_switch``
``StateInterface`` /         ``StateInterface`` / ``CommandInterface`` handles named ``<joint>/<interface>``
``CommandInterface``
``ResourceManager``          ``ResourceManager``: owns the components, the lifecycle, interface storage,
                             **exclusive** command-interface claims, command limits, error handling
``controller_manager``       ``ControllerManager`` (``controllers.hal.controllers``): read -> update -> write
lifecycle states             ``LifecycleState``: UNCONFIGURED, INACTIVE, ACTIVE, FINALIZED
``return_type``              ``ReturnType``: OK, ERROR
==========================  ===========================================================================

Interface names follow ``hardware_interface/types/hardware_interface_type_values.hpp``: ``position``,
``velocity``, ``effort``. Joint names are the MJCF joint names (``/nandi/joint_states`` of the HAL plan).

Hardware states: every component reports its lifecycle state and a status dict (``status()``: backend,
faults, per-joint active command mode), and the resource manager exposes them as ``component_states()``.
On a ``read``/``write`` that returns ERROR the component goes through ``on_error`` to UNCONFIGURED, its command
interfaces are released and the controllers that claimed them are deactivated (ros2_control Humble behaviour).
"""
import enum
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from atharv_nandi import paths

HW_IF_POSITION = "position"
HW_IF_VELOCITY = "velocity"
HW_IF_EFFORT = "effort"
JOINT_INTERFACES = (HW_IF_POSITION, HW_IF_VELOCITY, HW_IF_EFFORT)

REPO = paths.REPO
HAL_YAML = os.path.join(REPO, "config", "hal.yaml")
SCHEMA = "nandi.hal/1"


class ReturnType(enum.Enum):
    OK = 0
    ERROR = 1


class LifecycleState(enum.Enum):
    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FINALIZED = "finalized"


class HalError(RuntimeError):
    """Bad configuration, a lifecycle transition that is not allowed, or a claim conflict."""


# ---------------------------------------------------------------------- hardware info (config)
@dataclass
class InterfaceInfo:
    name: str
    min: float = -math.inf
    max: float = math.inf
    initial_value: float = 0.0


@dataclass
class ComponentInfo:
    """A joint or a sensor of a hardware component (ros2_control ``ComponentInfo``)."""
    name: str
    type: str = "joint"
    command_interfaces: List[InterfaceInfo] = field(default_factory=list)
    state_interfaces: List[InterfaceInfo] = field(default_factory=list)
    parameters: Dict[str, object] = field(default_factory=dict)

    def command(self, name: str) -> Optional[InterfaceInfo]:
        return next((i for i in self.command_interfaces if i.name == name), None)


@dataclass
class HardwareInfo:
    """One ``<ros2_control>`` block: a system with joints, sensors and parameters."""
    name: str
    type: str = "system"
    plugins: Dict[str, str] = field(default_factory=dict)
    joints: List[ComponentInfo] = field(default_factory=list)
    sensors: List[ComponentInfo] = field(default_factory=list)
    parameters: Dict[str, object] = field(default_factory=dict)

    def joint(self, name: str) -> ComponentInfo:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(name)

    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self.joints]


def _iface(name: str, spec) -> InterfaceInfo:
    spec = spec or {}
    return InterfaceInfo(name, float(spec.get("min", -math.inf)), float(spec.get("max", math.inf)),
                         float(spec.get("initial_value", 0.0)))


def _parse_component(spec: dict, name: str, kind: str) -> ComponentInfo:
    cmd = spec.get("command_interfaces", {}) or {}
    if isinstance(cmd, list):
        cmd = {c: {} for c in cmd}
    st = spec.get("state_interfaces", []) or []
    if isinstance(st, dict):
        st = list(st)
    for c in cmd:
        if kind == "joint" and c not in JOINT_INTERFACES:
            raise HalError(f"{name}: unknown command interface {c!r} (use {JOINT_INTERFACES})")
    params = {k: v for k, v in spec.items() if k not in ("name", "names", "command_interfaces", "state_interfaces")}
    return ComponentInfo(name, kind, [_iface(c, v) for c, v in cmd.items()], [InterfaceInfo(s) for s in st], params)


def parse_hardware(doc: dict) -> List[HardwareInfo]:
    """``HardwareInfo`` list from the ``hardware`` block of config/hal.yaml (``names:`` expands a template)."""
    if doc.get("schema") != SCHEMA:
        raise HalError(f"hal config schema must be {SCHEMA!r}, got {doc.get('schema')!r}")
    out = []
    for hw in doc.get("hardware", []) or []:
        if "name" not in hw:
            raise HalError("hardware block without a name")
        info = HardwareInfo(hw["name"], hw.get("type", "system"), dict(hw.get("plugins", {}) or {}),
                            parameters=dict(hw.get("parameters", {}) or {}))
        for kind, key in (("joint", "joints"), ("sensor", "sensors")):
            for spec in hw.get(key, []) or []:
                names = spec.get("names") or [spec.get("name")]
                for n in names:
                    if not n:
                        raise HalError(f"{info.name}: {key} entry without a name")
                    (info.joints if kind == "joint" else info.sensors).append(_parse_component(spec, n, kind))
        seen = set()
        for j in info.joints:
            if j.name in seen:
                raise HalError(f"{info.name}: joint {j.name} listed twice")
            seen.add(j.name)
            if not j.command_interfaces:
                raise HalError(f"{info.name}: joint {j.name} has no command interface")
        out.append(info)
    if not out:
        raise HalError("hal config has no hardware")
    return out


def load_config(path: Optional[str] = None) -> dict:
    import yaml
    with open(path or os.environ.get("NANDI_HAL_CONFIG", "") or HAL_YAML, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise HalError("hal config is not a mapping")
    return doc


def backend_name(doc: Optional[dict] = None) -> str:
    """Selected backend: env ``NANDI_HAL_BACKEND`` beats ``backend`` in config/hal.yaml (default mujoco)."""
    return (os.environ.get("NANDI_HAL_BACKEND", "").strip() or (doc or {}).get("backend") or "mujoco").lower()


# ---------------------------------------------------------------------- interface handles
class _Store:
    """Shared value storage of a resource manager, ``"<prefix>/<interface>" -> float``. State and command
    interfaces of the same name are distinct values (as in ros2_control), so they live in two maps."""

    def __init__(self):
        self.values: Dict[str, float] = {}      # state interfaces
        self.commands: Dict[str, float] = {}    # command interfaces


class StateInterface:
    def __init__(self, prefix: str, name: str, store: _Store):
        self.prefix, self.interface_name, self._store = prefix, name, store
        self.full_name = f"{prefix}/{name}"
        self._map().setdefault(self.full_name, math.nan)

    def _map(self) -> Dict[str, float]:
        return self._store.values

    def get_value(self) -> float:
        return self._map()[self.full_name]

    def set_value(self, v: float) -> None:        # hardware side only
        self._map()[self.full_name] = float(v)


class CommandInterface(StateInterface):
    """A command handle. Controllers get it only through ``ResourceManager.claim`` (exclusive)."""

    def __init__(self, prefix: str, name: str, store: _Store, info: Optional[InterfaceInfo] = None):
        super().__init__(prefix, name, store)
        self.info = info or InterfaceInfo(name)
        if not math.isnan(self.info.initial_value):
            store.commands[self.full_name] = self.info.initial_value

    def _map(self) -> Dict[str, float]:
        return self._store.commands

    def set_value(self, v: float) -> None:
        """Command, clamped to the interface's [min, max] (joint limits enforcement)."""
        v = float(v)
        if math.isnan(v):
            self._map()[self.full_name] = math.nan
            return
        self._map()[self.full_name] = min(self.info.max, max(self.info.min, v))


# ---------------------------------------------------------------------- system interface
class SystemInterface:
    """Base of a hardware component. Backends override the ``on_*`` callbacks, ``read`` and ``write``."""

    backend = "base"

    def __init__(self):
        self.info: Optional[HardwareInfo] = None
        self.state = LifecycleState.UNCONFIGURED
        self._store: Optional[_Store] = None
        self.state_ifaces: Dict[str, StateInterface] = {}
        self.command_ifaces: Dict[str, CommandInterface] = {}
        self.command_mode: Dict[str, str] = {}      # joint -> active command interface name
        self.faults: List[str] = []

    # lifecycle callbacks (return ReturnType) --------------------------------------------------------
    def on_init(self, info: HardwareInfo) -> ReturnType:
        self.info = info
        for j in info.joints:
            names = {c.name for c in j.command_interfaces}
            self.command_mode[j.name] = HW_IF_POSITION if HW_IF_POSITION in names else j.command_interfaces[0].name
        return ReturnType.OK

    def on_configure(self) -> ReturnType:
        return ReturnType.OK

    def on_activate(self) -> ReturnType:
        return ReturnType.OK

    def on_deactivate(self) -> ReturnType:
        return ReturnType.OK

    def on_cleanup(self) -> ReturnType:
        return ReturnType.OK

    def on_shutdown(self) -> ReturnType:
        return ReturnType.OK

    def on_error(self) -> ReturnType:
        return ReturnType.OK

    # interfaces ---------------------------------------------------------------------------------------
    def bind(self, store: _Store) -> None:
        self._store = store
        for comp in list(self.info.joints) + list(self.info.sensors):
            for si in comp.state_interfaces:
                h = StateInterface(comp.name, si.name, store)
                self.state_ifaces[h.full_name] = h
            for ci in comp.command_interfaces:
                h = CommandInterface(comp.name, ci.name, store, ci)
                self.command_ifaces[h.full_name] = h

    def export_state_interfaces(self) -> List[StateInterface]:
        return list(self.state_ifaces.values())

    def export_command_interfaces(self) -> List[CommandInterface]:
        return list(self.command_ifaces.values())

    def set_state(self, full_name: str, v: float) -> None:
        h = self.state_ifaces.get(full_name)
        if h is not None:
            h.set_value(v)

    def command(self, joint: str, iface: str) -> float:
        h = self.command_ifaces.get(f"{joint}/{iface}")
        return h.get_value() if h is not None else math.nan

    # command mode switch -------------------------------------------------------------------------------
    def prepare_command_mode_switch(self, start: Sequence[str], stop: Sequence[str]) -> ReturnType:
        """Refuse two command interfaces of one joint at once and interfaces this component does not own."""
        starting: Dict[str, str] = {}
        for full in start:
            if full not in self.command_ifaces:
                return ReturnType.ERROR
            joint, iface = full.rsplit("/", 1)
            if joint in starting:
                return ReturnType.ERROR
            starting[joint] = iface
        return ReturnType.OK

    def perform_command_mode_switch(self, start: Sequence[str], stop: Sequence[str]) -> ReturnType:
        for full in start:
            joint, iface = full.rsplit("/", 1)
            self.command_mode[joint] = iface
        return ReturnType.OK

    # loop ------------------------------------------------------------------------------------------------
    def read(self, time: float, period: float) -> ReturnType:
        raise NotImplementedError

    def write(self, time: float, period: float) -> ReturnType:
        raise NotImplementedError

    def status(self) -> dict:
        return {"name": self.info.name if self.info else "", "backend": self.backend, "state": self.state.value,
                "faults": list(self.faults), "command_mode": dict(self.command_mode)}


# ---------------------------------------------------------------------- resource manager
_TRANSITIONS = {
    "configure": ({LifecycleState.UNCONFIGURED}, "on_configure", LifecycleState.INACTIVE),
    "activate": ({LifecycleState.INACTIVE}, "on_activate", LifecycleState.ACTIVE),
    "deactivate": ({LifecycleState.ACTIVE}, "on_deactivate", LifecycleState.INACTIVE),
    "cleanup": ({LifecycleState.INACTIVE}, "on_cleanup", LifecycleState.UNCONFIGURED),
    "shutdown": ({LifecycleState.UNCONFIGURED, LifecycleState.INACTIVE, LifecycleState.ACTIVE}, "on_shutdown",
                 LifecycleState.FINALIZED),
}


class ResourceManager:
    """Owns hardware components, their interface storage, lifecycle and command-interface claims."""

    def __init__(self):
        self.store = _Store()
        self.components: Dict[str, SystemInterface] = {}
        self._claims: Dict[str, str] = {}               # full command interface name -> owner
        self.on_component_error: List[Callable[[str], None]] = []

    def add(self, component: SystemInterface, info: HardwareInfo) -> SystemInterface:
        if info.name in self.components:
            raise HalError(f"component {info.name} already loaded")
        for other in self.components.values():
            clash = set(other.info.joint_names) & set(info.joint_names)
            if clash:
                raise HalError(f"joints {sorted(clash)} already owned by {other.info.name}")
        if component.on_init(info) != ReturnType.OK:
            raise HalError(f"{info.name}: on_init failed")
        component.bind(self.store)
        self.components[info.name] = component
        return component

    # lifecycle -----------------------------------------------------------------------------------------
    def transition(self, name: str, action: str) -> LifecycleState:
        comp = self.components[name]
        allowed, cb, target = _TRANSITIONS[action]
        if comp.state not in allowed:
            raise HalError(f"{name}: cannot {action} from {comp.state.value}")
        if action == "shutdown" and comp.state == LifecycleState.ACTIVE:
            self.transition(name, "deactivate")
        if action == "deactivate":
            self._release_component(name)
        if getattr(comp, cb)() != ReturnType.OK:
            self._fail(name)
            raise HalError(f"{name}: {cb} failed")
        comp.state = target
        return target

    def configure_all(self) -> None:
        for n in self.components:
            self.transition(n, "configure")

    def activate_all(self) -> None:
        for n, c in self.components.items():
            if c.state == LifecycleState.UNCONFIGURED:
                self.transition(n, "configure")
            self.transition(n, "activate")

    def _fail(self, name: str) -> None:
        comp = self.components[name]
        self._release_component(name)
        comp.on_error()
        comp.state = LifecycleState.UNCONFIGURED
        for cb in self.on_component_error:
            cb(name)

    # interfaces ---------------------------------------------------------------------------------------
    def state_interface_names(self) -> List[str]:
        return [n for c in self.components.values() for n in c.state_ifaces]

    def command_interface_names(self) -> List[str]:
        return [n for c in self.components.values() for n in c.command_ifaces]

    def _owner(self, full: str) -> SystemInterface:
        for c in self.components.values():
            if full in c.command_ifaces or full in c.state_ifaces:
                return c
        raise KeyError(full)

    def state(self, full: str) -> float:
        return self.store.values[full]

    def command_value(self, full: str) -> float:
        return self.store.commands[full]

    def is_available(self, full: str) -> bool:
        try:
            return self._owner(full).state == LifecycleState.ACTIVE
        except KeyError:
            return False

    def claim(self, names: Sequence[str], owner: str) -> List[CommandInterface]:
        """Exclusive claim of command interfaces (all or none); switches the joints' command mode."""
        for n in names:
            if n in self._claims and self._claims[n] != owner:
                raise HalError(f"{n} already claimed by {self._claims[n]}")
            if not self.is_available(n) or n not in self._owner(n).command_ifaces:
                raise HalError(f"{n} is not an available command interface")
        by_comp: Dict[str, List[str]] = {}
        for n in names:
            by_comp.setdefault(self._owner(n).info.name, []).append(n)
        for cname, start in by_comp.items():
            comp = self.components[cname]
            # a joint may be commanded through one interface at a time, across all owners
            for n in start:
                joint = n.rsplit("/", 1)[0]
                held = [c for c in self._claims if c.rsplit("/", 1)[0] == joint and c != n]
                if held:
                    raise HalError(f"{joint} is already commanded through {held[0]}")
            if comp.prepare_command_mode_switch(start, []) != ReturnType.OK:
                raise HalError(f"{cname}: command mode switch to {start} refused")
        for cname, start in by_comp.items():
            self.components[cname].perform_command_mode_switch(start, [])
        for n in names:
            self._claims[n] = owner
        return [self._owner(n).command_ifaces[n] for n in names]

    def release(self, owner: str) -> None:
        """Release every claim of ``owner`` and leave its joints in a safe hold (safety interlock).

        A released velocity / effort interface must not keep acting on its last command: its value is zeroed and
        the joint switches back to position with target = current position (bumpless hold, servo back on), or,
        when the joint has no position interface, re-enters its own mode with a zero command. A released
        position interface keeps holding its last target (the servo already holds it)."""
        names = [n for n, o in self._claims.items() if o == owner]
        for n in names:
            del self._claims[n]
        for n in names:
            joint, iface = n.rsplit("/", 1)
            if iface == HW_IF_POSITION:
                continue
            comp = self._owner(n)
            comp.command_ifaces[n].set_value(0.0)
            hold = f"{joint}/{HW_IF_POSITION}"
            if hold in comp.command_ifaces:
                comp.perform_command_mode_switch([hold], [n])
            else:
                comp.perform_command_mode_switch([n], [n])

    def _release_component(self, name: str) -> None:
        comp = self.components[name]
        for n in [n for n in self._claims if n in comp.command_ifaces]:
            del self._claims[n]

    def claimed(self) -> Dict[str, str]:
        return dict(self._claims)

    # loop ----------------------------------------------------------------------------------------------
    def read(self, time: float, period: float) -> Dict[str, ReturnType]:
        return self._loop("read", time, period)

    def write(self, time: float, period: float) -> Dict[str, ReturnType]:
        return self._loop("write", time, period)

    def _loop(self, what: str, time: float, period: float) -> Dict[str, ReturnType]:
        out = {}
        for n, c in self.components.items():
            if c.state not in (LifecycleState.ACTIVE, LifecycleState.INACTIVE):
                continue
            if what == "write" and c.state != LifecycleState.ACTIVE:
                continue
            r = getattr(c, what)(time, period)
            out[n] = r
            if r != ReturnType.OK:
                self._fail(n)
        return out

    def component_states(self) -> Dict[str, dict]:
        return {n: c.status() for n, c in self.components.items()}


def load_plugin(path: str):
    """Class from ``"package.module.Class"`` (the ``plugins`` entry of a hardware block)."""
    import importlib
    mod, _, cls = path.rpartition(".")
    return getattr(importlib.import_module(mod), cls)


def build_resource_manager(doc: Optional[dict] = None, backend: Optional[str] = None,
                           factory_kwargs: Optional[dict] = None,
                           hardware: Optional[Iterable[str]] = None) -> ResourceManager:
    """Resource manager with every hardware block of the config loaded with the selected backend's plugin.

    ``factory_kwargs`` go to the plugin constructor (``model`` / ``data`` for the MuJoCo backend).
    """
    doc = doc if doc is not None else load_config()
    backend = backend or backend_name(doc)
    rm = ResourceManager()
    for info in parse_hardware(doc):
        if hardware is not None and info.name not in hardware:
            continue
        if backend not in info.plugins:
            raise HalError(f"{info.name}: no plugin for backend {backend!r} (have {sorted(info.plugins)})")
        cls = load_plugin(info.plugins[backend])
        rm.add(cls(**(factory_kwargs or {})), info)
    return rm


def joint_limits(doc: dict, joints: Sequence[str]) -> Tuple[List[float], List[float]]:
    """(max_velocity, max_acceleration) per joint from the ``limits`` of each joint in the hal config."""
    v, a = {}, {}
    for info in parse_hardware(doc):
        for j in info.joints:
            lim = j.parameters.get("limits", {}) or {}
            v[j.name] = float(lim.get("max_velocity", math.inf))
            a[j.name] = float(lim.get("max_acceleration", math.inf))
    missing = [j for j in joints if j not in v]
    if missing:
        raise HalError(f"no limits for {missing}")
    return [v[j] for j in joints], [a[j] for j in joints]
