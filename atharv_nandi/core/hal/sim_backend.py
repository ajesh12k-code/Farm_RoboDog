"""MuJoCo backend of the HAL (WPCAP): a ``SystemInterface`` over a live ``MjModel`` / ``MjData``.

The sim's actuators are MuJoCo ``position`` servos (kp/kv, forcerange; atharv_nandi.xml). The three ros2_control
command interfaces map onto them like this:

==========  =================================================================================================
position    ``ctrl[act] = command`` (+ optional gravity feed-forward ``qfrc_bias[dof] / kp`` per joint,
            parameter ``gravity_feedforward``: the same correction arm_ik_solver.gravity_offsets applies)
velocity    the position target integrates the command: ``target += v * period`` (clamped to ctrlrange),
            ``ctrl[act] = target``. A position servo has no velocity reference, so a ramp lags by kv*v/kp.
effort      the joint's servo is switched off (gain and bias of that actuator set to 0 on the **live model**,
            restored when the joint leaves effort mode or the component deactivates) and the command is
            written to ``qfrc_applied[dof]``, clamped to the actuator forcerange
==========  =================================================================================================

State interfaces: ``position`` = qpos, ``velocity`` = qvel, ``effort`` = qfrc_actuator + qfrc_applied at the dof.
Sensor components read named MuJoCo sensors (``parameters.sources``: ``{"orientation": "imu_quat", ...}``); a
state interface ``orientation.x`` is component ``x`` of that sensor (quaternions w, x, y, z; vectors x, y, z).

``read`` / ``write`` never step physics: the sim driver (or a test) calls ``mj_step`` between them.
"""
import math
from typing import Dict, Optional

import mujoco
import numpy as np

from atharv_nandi.core.hal.interfaces import (
    HW_IF_EFFORT, HW_IF_POSITION, HW_IF_VELOCITY, HardwareInfo, ReturnType, SystemInterface,
)

_AXES = {4: ("w", "x", "y", "z"), 3: ("x", "y", "z"), 1: ("",)}


class MujocoSystem(SystemInterface):
    backend = "mujoco"

    def __init__(self, model: mujoco.MjModel = None, data: mujoco.MjData = None, **_):
        super().__init__()
        if model is None or data is None:
            raise ValueError("MujocoSystem needs model= and data=")
        self.m, self.d = model, data
        self._j: Dict[str, dict] = {}
        self._sensor_src: Dict[str, tuple] = {}
        self._saved_gain: Dict[int, tuple] = {}

    def on_init(self, info: HardwareInfo) -> ReturnType:
        if super().on_init(info) != ReturnType.OK:
            return ReturnType.ERROR
        m = self.m
        for j in info.joints:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j.name)
            if jid < 0:
                return ReturnType.ERROR
            acts = [a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                    and m.actuator_trnid[a, 0] == jid]
            if not acts:
                return ReturnType.ERROR
            a = acts[0]
            self._j[j.name] = {"jid": jid, "qadr": int(m.jnt_qposadr[jid]), "dof": int(m.jnt_dofadr[jid]),
                               "act": a, "kp": float(m.actuator_gainprm[a, 0]),
                               "ctrlrange": tuple(float(v) for v in m.actuator_ctrlrange[a]),
                               "forcerange": tuple(float(v) for v in m.actuator_forcerange[a]),
                               "ff": bool(j.parameters.get("gravity_feedforward", False)), "target": math.nan}
        for s in info.sensors:
            for iface, sname in (s.parameters.get("sources", {}) or {}).items():
                sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sname)
                if sid < 0:
                    return ReturnType.ERROR
                self._sensor_src[f"{s.name}/{iface}"] = (int(m.sensor_adr[sid]), int(m.sensor_dim[sid]))
        return ReturnType.OK

    # lifecycle -----------------------------------------------------------------------------------------
    def on_configure(self) -> ReturnType:
        self.read(0.0, 0.0)
        return ReturnType.OK

    def on_activate(self) -> ReturnType:
        # hold the current pose: command = state, so activation never jumps the robot
        for name, j in self._j.items():
            q = float(self.d.qpos[j["qadr"]])
            j["target"] = q
            for iface, v in ((HW_IF_POSITION, q), (HW_IF_VELOCITY, 0.0), (HW_IF_EFFORT, 0.0)):
                h = self.command_ifaces.get(f"{name}/{iface}")
                if h is not None:
                    h.set_value(v)
        return ReturnType.OK

    def on_deactivate(self) -> ReturnType:
        for name in list(self._j):
            self._leave_effort(name)
            self.command_mode[name] = HW_IF_POSITION
        return ReturnType.OK

    def on_error(self) -> ReturnType:
        return self.on_deactivate()

    # mode switch ---------------------------------------------------------------------------------------
    def perform_command_mode_switch(self, start, stop) -> ReturnType:
        for full in start:
            joint, iface = full.rsplit("/", 1)
            j = self._j[joint]
            if self.command_mode.get(joint) == HW_IF_EFFORT and iface != HW_IF_EFFORT:
                self._leave_effort(joint)
            if iface == HW_IF_EFFORT:
                self._enter_effort(joint)
            j["target"] = float(self.d.qpos[j["qadr"]])
            # bumpless: the newly claimed interface starts from the current state
            h = self.command_ifaces[full]
            h.set_value(j["target"] if iface == HW_IF_POSITION else 0.0)
            self.command_mode[joint] = iface
        return ReturnType.OK

    def _enter_effort(self, joint: str) -> None:
        a = self._j[joint]["act"]
        if a not in self._saved_gain:
            self._saved_gain[a] = (self.m.actuator_gainprm[a].copy(), self.m.actuator_biasprm[a].copy())
        self.m.actuator_gainprm[a, 0] = 0.0
        self.m.actuator_biasprm[a, 1:3] = 0.0

    def _leave_effort(self, joint: str) -> None:
        j = self._j[joint]
        a = j["act"]
        if a in self._saved_gain:
            g, b = self._saved_gain.pop(a)
            self.m.actuator_gainprm[a] = g
            self.m.actuator_biasprm[a] = b
        self.d.qfrc_applied[j["dof"]] = 0.0

    # loop ----------------------------------------------------------------------------------------------
    def read(self, time: float, period: float) -> ReturnType:
        d = self.d
        for name, j in self._j.items():
            self.set_state(f"{name}/{HW_IF_POSITION}", d.qpos[j["qadr"]])
            self.set_state(f"{name}/{HW_IF_VELOCITY}", d.qvel[j["dof"]])
            self.set_state(f"{name}/{HW_IF_EFFORT}", d.qfrc_actuator[j["dof"]] + d.qfrc_applied[j["dof"]])
        for full, (adr, dim) in self._sensor_src.items():
            for k, ax in enumerate(_AXES.get(dim, tuple(str(i) for i in range(dim)))):
                self.set_state(f"{full}.{ax}" if ax else full, d.sensordata[adr + k])
        return ReturnType.OK

    def write(self, time: float, period: float) -> ReturnType:
        d = self.d
        for name, j in self._j.items():
            mode = self.command_mode[name]
            cmd = self.command(name, mode)
            lo, hi = j["ctrlrange"]
            if mode == HW_IF_EFFORT:
                flo, fhi = j["forcerange"]
                d.qfrc_applied[j["dof"]] = 0.0 if math.isnan(cmd) else min(fhi, max(flo, cmd))
                continue
            if mode == HW_IF_VELOCITY:
                if not math.isnan(cmd):
                    j["target"] = min(hi, max(lo, j["target"] + cmd * period))
                tgt = j["target"]
            else:
                tgt = j["target"] if math.isnan(cmd) else cmd
                j["target"] = tgt
            ff = float(d.qfrc_bias[j["dof"]]) / j["kp"] if j["ff"] and j["kp"] > 0 else 0.0
            d.ctrl[j["act"]] = min(hi, max(lo, tgt + ff))
        return ReturnType.OK

    def status(self) -> dict:
        s = super().status()
        s["effort_mode_actuators"] = sorted(self._saved_gain)
        return s
