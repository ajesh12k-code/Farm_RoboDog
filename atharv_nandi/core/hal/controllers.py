"""ros2_control-style controllers and controller manager over a ``ResourceManager`` (WPCAP).

* ``JointStateBroadcaster``     reads every joint's position / velocity / effort state interfaces and returns a
                                ``sensor_msgs/JointState``-shaped dict (``name``, ``position``, ``velocity``,
                                ``effort``); claims nothing.
* ``ForwardCommandController``  forwards a command vector to one interface type of a joint list
                                (``position_controllers/JointGroupPositionController`` and friends).
* ``JointTrajectoryController``  executes a ``controllers.planning.Trajectory`` (or any object with
                                ``sample(t)`` / ``duration``) through position interfaces and reports the
                                tracking error (``joint_trajectory_controller`` with a stop-at-goal tolerance).
                                A trajectory whose first point is farther than ``start_tolerance`` from the
                                current joint state is refused (``HalError``), so the servos never step to it.
* ``ControllerManager``         ``update(time, period)`` = resource manager ``read`` -> every active controller's
                                ``update`` -> ``write``. Activating a controller claims its command interfaces
                                exclusively; a hardware error deactivates every controller that held one of the
                                failed component's interfaces. Deactivating releases the claims and the resource
                                manager puts velocity / effort joints back into a position hold at the current
                                pose (zero command), so nothing keeps moving after its controller stopped.
"""
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from atharv_nandi.core.hal.interfaces import (
    HW_IF_EFFORT, HW_IF_POSITION, HW_IF_VELOCITY, HalError, ResourceManager,
)


class Controller:
    def __init__(self, name: str):
        self.name = name
        self.active = False
        self.rm: Optional[ResourceManager] = None
        self.handles = []

    def command_interfaces(self) -> List[str]:
        return []

    def on_activate(self) -> None:
        pass

    def update(self, time: float, period: float) -> None:
        raise NotImplementedError


class JointStateBroadcaster(Controller):
    def __init__(self, name: str = "joint_state_broadcaster", joints: Optional[Sequence[str]] = None):
        super().__init__(name)
        self.joints = list(joints) if joints else None
        self.msg: Dict[str, list] = {"name": [], "position": [], "velocity": [], "effort": []}

    def update(self, time: float, period: float) -> None:
        names = self.joints
        if names is None:
            names = []
            for c in self.rm.components.values():
                names.extend(c.info.joint_names)
        msg = {"stamp": time, "name": list(names), "position": [], "velocity": [], "effort": []}
        for j in names:
            for k, iface in (("position", HW_IF_POSITION), ("velocity", HW_IF_VELOCITY), ("effort", HW_IF_EFFORT)):
                full = f"{j}/{iface}"
                msg[k].append(self.rm.state(full) if full in self.rm.store.values else math.nan)
        self.msg = msg


class ForwardCommandController(Controller):
    def __init__(self, name: str, joints: Sequence[str], interface: str = HW_IF_POSITION):
        super().__init__(name)
        self.joints, self.interface = list(joints), interface
        self.commands: Optional[np.ndarray] = None

    def command_interfaces(self) -> List[str]:
        return [f"{j}/{self.interface}" for j in self.joints]

    def on_activate(self) -> None:
        self.commands = None          # no stale command from a previous activation (ros2 resets it too)

    def set_command(self, values: Sequence[float]) -> None:
        if len(values) != len(self.joints):
            raise ValueError(f"{self.name}: expected {len(self.joints)} values")
        self.commands = np.asarray(values, dtype=np.float64)

    def update(self, time: float, period: float) -> None:
        if self.commands is None:
            return
        for h, v in zip(self.handles, self.commands):
            h.set_value(v)


class JointTrajectoryController(Controller):
    def __init__(self, name: str, joints: Sequence[str], goal_tolerance: float = 0.01, start_tolerance: float = 0.05):
        super().__init__(name)
        self.joints = list(joints)
        self.goal_tolerance = float(goal_tolerance)
        self.start_tolerance = float(start_tolerance)
        self.trajectory = None
        self.t_start = None
        self.max_error = 0.0
        self.done = True

    def command_interfaces(self) -> List[str]:
        return [f"{j}/{HW_IF_POSITION}" for j in self.joints]

    def follow(self, trajectory) -> None:
        """Start a trajectory; refused if its first point is not within ``start_tolerance`` of the state."""
        if self.rm is not None and self.start_tolerance > 0.0:
            q = self.positions()
            q0 = np.asarray(trajectory.sample(0.0), dtype=np.float64)
            if np.all(np.isfinite(q)):
                err = float(np.max(np.abs(q - q0)))
                if err > self.start_tolerance:
                    raise HalError(f"{self.name}: trajectory starts {err:.3f} from the current state "
                                   f"(start_tolerance {self.start_tolerance})")
        self.trajectory, self.t_start, self.max_error, self.done = trajectory, None, 0.0, False

    def on_activate(self) -> None:
        self.trajectory, self.t_start, self.done = None, None, True

    def positions(self) -> np.ndarray:
        return np.array([self.rm.state(f"{j}/{HW_IF_POSITION}") for j in self.joints])

    def update(self, time: float, period: float) -> None:
        if self.trajectory is None or self.done:
            return
        if self.t_start is None:
            self.t_start = time
        t = time - self.t_start
        ref = self.trajectory.sample(t)
        for h, v in zip(self.handles, ref):
            h.set_value(v)
        err = float(np.max(np.abs(self.positions() - self.trajectory.sample(max(0.0, t - period)))))
        self.max_error = max(self.max_error, err)
        if t >= self.trajectory.duration:
            final = float(np.max(np.abs(self.positions() - self.trajectory.sample(self.trajectory.duration))))
            self.done = final <= self.goal_tolerance


class ControllerManager:
    def __init__(self, rm: ResourceManager, update_rate_hz: float = 100.0):
        self.rm = rm
        self.update_rate_hz = float(update_rate_hz)
        self.controllers: Dict[str, Controller] = {}
        rm.on_component_error.append(self._on_hw_error)
        self.events: List[str] = []

    def add(self, c: Controller) -> Controller:
        if c.name in self.controllers:
            raise HalError(f"controller {c.name} already loaded")
        c.rm = self.rm
        self.controllers[c.name] = c
        return c

    def activate(self, name: str) -> None:
        c = self.controllers[name]
        c.handles = self.rm.claim(c.command_interfaces(), owner=name) if c.command_interfaces() else []
        c.active = True
        c.on_activate()

    def deactivate(self, name: str) -> None:
        c = self.controllers[name]
        self.rm.release(name)
        c.handles, c.active = [], False

    def switch(self, stop: Sequence[str] = (), start: Sequence[str] = ()) -> None:
        for n in stop:
            self.deactivate(n)
        for n in start:
            self.activate(n)

    def _on_hw_error(self, component: str) -> None:
        names = set(self.rm.components[component].command_ifaces)
        for c in self.controllers.values():
            if c.active and names & set(c.command_interfaces()):
                c.handles, c.active = [], False
                self.events.append(f"deactivated {c.name}: hardware {component} error")

    def update(self, time: float, period: float) -> Dict[str, object]:
        r = self.rm.read(time, period)
        for c in self.controllers.values():
            if c.active:
                c.update(time, period)
        w = self.rm.write(time, period)
        return {"read": r, "write": w}
