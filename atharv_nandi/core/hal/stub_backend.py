"""Stub hardware backend of the HAL (WPCAP): no MuJoCo, no bus, deterministic.

Stands in for the real servo-bus driver on the bench path and in ROS-free tests of anything above the HAL
(controllers, the planner's trajectory execution, error handling). Per joint it integrates a simple servo:

* position mode: first-order lag toward the command (time constant ``tau_s``, default 0.03 s, the HAL plan's
  ASSUMED joint-command lag) with the joint's ``max_velocity`` limit;
* velocity mode: the commanded velocity (clamped to ``max_velocity``), first-order lag ``tau_s``;
* effort mode: a damped inertia ``J qdd = u - b qd`` (parameters ``inertia``, default 0.05, ``damping``, 0.5);

positions are clamped to the position command interface's [min, max] (the joint range). Optional transport delay
(``delay_steps``) and Gaussian position noise (``noise_std`` with ``seed``) mirror the noise and latency layer of
docs/hal-sim-to-real.md section 4. Faults: ``inject_fault("comm_timeout")`` makes the next ``read`` return
ERROR (the resource manager then runs ``on_error`` and the component drops to UNCONFIGURED);
``inject_fault("overcurrent:<joint>")`` makes ``write`` return ERROR.
"""
import math
from collections import deque
from typing import Dict

import numpy as np

from atharv_nandi.core.hal.interfaces import (
    HW_IF_EFFORT, HW_IF_POSITION, HW_IF_VELOCITY, HardwareInfo, ReturnType, SystemInterface,
)


class StubSystem(SystemInterface):
    backend = "stub"

    def __init__(self, tau_s: float = 0.03, delay_steps: int = 0, noise_std: float = 0.0, seed: int = 7,
                 initial_positions: Dict[str, float] = None, **_):
        super().__init__()
        self.tau_s, self.delay_steps, self.noise_std = float(tau_s), int(delay_steps), float(noise_std)
        self.rng = np.random.default_rng(seed)
        self.q: Dict[str, float] = {}
        self.qd: Dict[str, float] = {}
        self.u: Dict[str, float] = {}
        self._init = dict(initial_positions or {})
        self._pending: deque = deque()
        self._faults_armed: set = set()

    def on_init(self, info: HardwareInfo) -> ReturnType:
        if super().on_init(info) != ReturnType.OK:
            return ReturnType.ERROR
        for j in info.joints:
            self.q[j.name] = float(self._init.get(j.name, j.parameters.get("initial_position", 0.0)))
            self.qd[j.name] = 0.0
            self.u[j.name] = 0.0
        return ReturnType.OK

    def on_configure(self) -> ReturnType:
        self.faults.clear()
        self._pending.clear()
        self.read(0.0, 0.0)
        return ReturnType.OK

    def on_activate(self) -> ReturnType:
        for name in self.q:
            for iface, v in ((HW_IF_POSITION, self.q[name]), (HW_IF_VELOCITY, 0.0), (HW_IF_EFFORT, 0.0)):
                h = self.command_ifaces.get(f"{name}/{iface}")
                if h is not None:
                    h.set_value(v)
        return ReturnType.OK

    def on_error(self) -> ReturnType:
        for name in self.qd:
            self.qd[name] = 0.0
        return ReturnType.OK

    def perform_command_mode_switch(self, start, stop) -> ReturnType:
        for full in start:
            joint, iface = full.rsplit("/", 1)
            self.command_ifaces[full].set_value(self.q[joint] if iface == HW_IF_POSITION else 0.0)
            self.command_mode[joint] = iface
        return ReturnType.OK

    def inject_fault(self, fault: str) -> None:
        self._faults_armed.add(fault)

    # ---------------------------------------------------------------------------------------------------
    def _limits(self, joint: str):
        info = self.info.joint(joint)
        pos = info.command(HW_IF_POSITION)
        lim = info.parameters.get("limits", {}) or {}
        vmax = float(lim.get("max_velocity", math.inf))
        lo, hi = (pos.min, pos.max) if pos is not None else (-math.inf, math.inf)
        return lo, hi, vmax

    def read(self, time: float, period: float) -> ReturnType:
        if "comm_timeout" in self._faults_armed:
            self._faults_armed.discard("comm_timeout")
            self.faults.append("comm_timeout")
            return ReturnType.ERROR
        for name in self.q:
            noise = self.rng.normal(0.0, self.noise_std) if self.noise_std > 0 else 0.0
            self.set_state(f"{name}/{HW_IF_POSITION}", self.q[name] + noise)
            self.set_state(f"{name}/{HW_IF_VELOCITY}", self.qd[name])
            self.set_state(f"{name}/{HW_IF_EFFORT}", self.u[name])
        return ReturnType.OK

    def write(self, time: float, period: float) -> ReturnType:
        for f in [f for f in self._faults_armed if f.startswith("overcurrent")]:
            self._faults_armed.discard(f)
            self.faults.append(f)
            return ReturnType.ERROR
        cmds = {n: (self.command_mode[n], self.command(n, self.command_mode[n])) for n in self.q}
        self._pending.append(cmds)
        if len(self._pending) <= self.delay_steps:
            return ReturnType.OK
        cmds = self._pending.popleft()
        if period <= 0.0:
            return ReturnType.OK
        alpha = 1.0 - math.exp(-period / self.tau_s) if self.tau_s > 0 else 1.0
        for name, (mode, cmd) in cmds.items():
            lo, hi, vmax = self._limits(name)
            info = self.info.joint(name)
            if math.isnan(cmd):
                cmd = self.q[name] if mode == HW_IF_POSITION else 0.0
            if mode == HW_IF_POSITION:
                v = (cmd - self.q[name]) * alpha / period
                v = max(-vmax, min(vmax, v))
                self.u[name] = 0.0
            elif mode == HW_IF_VELOCITY:
                cmd = max(-vmax, min(vmax, cmd))
                v = self.qd[name] + (cmd - self.qd[name]) * alpha
                self.u[name] = 0.0
            else:
                J = float(info.parameters.get("inertia", 0.05))
                b = float(info.parameters.get("damping", 0.5))
                self.u[name] = cmd
                v = self.qd[name] + (cmd - b * self.qd[name]) / J * period
            q = self.q[name] + v * period
            if q < lo or q > hi:
                q, v = min(hi, max(lo, q)), 0.0
            self.q[name], self.qd[name] = q, v
        return ReturnType.OK
