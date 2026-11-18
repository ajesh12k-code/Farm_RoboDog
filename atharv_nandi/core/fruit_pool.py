"""Fruit proxy pool runtime (WP23; ORCHARD-MVP-SPEC 3.4). ROS-free; MuJoCo + numpy.

Trees are static (design rule 13). When a tree's HARVEST starts, every target fruit gets a pool body:

    assign(d, fruit_id)   hide the static geom (alpha 0, ``hidden_geoms`` provider); reshape the slot body to the
                          fruit (size, mass, inertia, pedicel capsule, ``cut_k`` / stem-end / anchor sites);
                          teleport it; write ``eq_data`` (connect: anchor in the fruit frame + world anchor at the
                          pedicel top; weld: anchor at the fruit CoM in the world frame, relpose, torquescale 1);
                          activate the equality to the world
    release(d, slot)      equality off: the logical cut (mango clip) or the twist release; the fruit is free
    recycle(d, slot, harvested)   park the body, restore the parked template, re-hold it; the static geom comes
                          back unless the fruit was harvested (then it stays hidden)
    reset(d)              recycle every slot and un-hide every static (world reset)

Slot kinds are fixed (0-4 connect for mango, 5-9 weld for guava / papaya); a fruit whose crop profile says
``pool_kind: none`` (banana) is refused. Measurements for the task layers (WP2 cut rule and twist logic, WP8):
``cut_point`` / ``stem_end`` / ``anchor_error_m``, ``pedicel_error_mm`` (|cut point to stem end - setpoint| in
the fruit frame), ``reaction`` (the equality's force and torque on the fruit, world frame), ``stem_axis_torque``
and ``angular_rate_dps``. ``TwistReleaseStat`` is the reference form of the OS 3.4 release statistic used by
the tau sweep (``python3 -m atharv_nandi.core.fruit_pool --sweep``); WP2 ``twist_logic`` is its runtime owner.
"""
import json
import math
import os
import sys
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from atharv_nandi.sim.world.modules import fruit_pool as pool_mod  # noqa: E402
from atharv_nandi.sim.world.modules import trees as trees_mod  # noqa: E402
from atharv_nandi.sim.world import trees as tree_gen  # noqa: E402


class PoolError(RuntimeError):
    pass


def _quat_conj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def _capsule_fields(a, b, r: float):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pos, quat, half = tree_gen.capsule_frame(a, b)
    half = max(half, 1e-4)
    return pos, quat, [r, half, 0.0], half + r, [0.0, 0.0, 0.0, r, r, half + r]


class FruitPool:
    """Assign / release / recycle the 10 pool slots of a composed orchard world."""

    def __init__(self, m: mujoco.MjModel, plans: Optional[Sequence[dict]] = None):
        st = pool_mod.model_state(m)
        if st is None:
            raise PoolError("this world has no fruit pool (no orchard trees)")
        self.m = m
        self._st = st
        self.slots = st["slots"]
        if plans is not None:
            self.plans = list(plans)
        else:
            self.plans = trees_mod.plans_for_model(m)
            if not self.plans:
                raise PoolError("the plans of this model are unknown here: pass plans= (world.trees.world_plans)")
        self.fruit = tree_gen.fruit_table(self.plans)
        self._static_gid = {}
        for fid, f in self.fruit.items():
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f["geom"])
            if gid < 0:
                raise PoolError(f"static geom {f['geom']} of {fid} is not in this model (plans of another world?)")
            self._static_gid[fid] = gid
        if plans is not None and trees_mod.plans_match(m, self.plans):
            trees_mod.bind_plans(m, self.plans)
        self._park = st["park"]           # compiled parked template (captured before any write)

    # -------------------------------------------------------------- queries
    @property
    def assignments(self) -> Dict[int, dict]:
        return {k: dict(v) for k, v in self._st["assign"].items()}

    @property
    def harvested(self) -> List[str]:
        return sorted(self._st["harvested"])

    def slot_of(self, fruit_id: str) -> Optional[int]:
        for k, a in self._st["assign"].items():
            if a["fruit_id"] == fruit_id:
                return k
        return None

    def free_slots(self, kind: str) -> List[int]:
        return [s.k for s in self.slots if s.kind == kind and s.k not in self._st["assign"]]

    def static_geom(self, fruit_id: str) -> int:
        return self._static_gid[fruit_id]

    # -------------------------------------------------------------- assign / release / recycle
    def _check_assignable(self, fruit_id: str) -> str:
        """The fruit's slot kind, or PoolError (unknown, MARK-only crop, bagged, already assigned / harvested)."""
        f = self.fruit.get(fruit_id)
        if f is None:
            raise PoolError(f"unknown fruit {fruit_id!r}")
        kind = f["pool_kind"]
        if kind not in ("connect", "weld"):
            raise PoolError(f"{fruit_id}: crop {f['crop']} has no pool kind ({kind}); MARK only")
        if f.get("bagged"):
            raise PoolError(f"{fruit_id}: bagged fruit are not harvested (reason bagged)")
        if self.slot_of(fruit_id) is not None or fruit_id in self._st["harvested"]:
            raise PoolError(f"{fruit_id}: already assigned or harvested")
        return kind

    def assign(self, d: mujoco.MjData, fruit_id: str, forward: bool = True) -> int:
        """Give ``fruit_id`` a pool body of its kind; returns the slot. Raises PoolError (no slot, wrong kind,
        already assigned or harvested, bagged) before writing anything."""
        kind = self._check_assignable(fruit_id)
        f = self.fruit[fruit_id]
        free = self.free_slots(kind)
        if not free:
            raise PoolError(f"no free {kind} slot (5 per kind; recycle one first)")
        s = self.slots[free[0]]
        m = self.m
        centre = np.asarray(f["centre_world"], float)
        quat = np.array([1.0, 0.0, 0.0, 0.0])       # fruit frame = world axes, z = stem axis (tree frame)
        # body: shape, mass, inertia, invweights (constraint scaling approximations of a lone free body)
        size = np.asarray(f["size_m"], float)
        mass = float(f["mass_kg"])
        inertia = pool_mod.ellipsoid_inertia(mass, size)
        m.geom_size[s.fruit] = size
        m.geom_rgba[s.fruit] = f["rgba"]
        m.geom_contype[s.fruit], m.geom_conaffinity[s.fruit] = 1, 0
        m.body_contype[s.body], m.body_conaffinity[s.body] = 1, 0     # MuJoCo filters bodies first (OR of geoms)
        m.geom_rbound[s.fruit] = float(np.max(size))
        m.geom_aabb[s.fruit] = [0.0, 0.0, 0.0, *size]
        m.body_mass[s.body] = mass
        m.body_subtreemass[s.body] = mass
        m.body_inertia[s.body] = inertia
        m.body_invweight0[s.body] = [1.0 / mass, 3.0 / float(sum(inertia))]
        m.dof_invweight0[s.dadr:s.dadr + 3] = 1.0 / mass
        m.dof_invweight0[s.dadr + 3:s.dadr + 6] = [1.0 / i for i in inertia]
        pool_mod.sync_dof_m0(m, s)       # simple body: MuJoCo's mass matrix is dof_M0, not body_mass
        # pedicel capsule + sites, in the fruit frame
        stem = np.asarray(f["stem_end_world"], float) - centre
        top = np.asarray(f["anchor_top_world"], float) - centre
        cut = np.asarray(f["cut_k_world"], float) - centre
        pos, pq, psize, prb, paabb = _capsule_fields(stem, top, 0.0025)
        m.geom_pos[s.pedicel], m.geom_quat[s.pedicel], m.geom_size[s.pedicel] = pos, pq, psize
        m.geom_rbound[s.pedicel], m.geom_aabb[s.pedicel] = prb, paabb
        m.geom_rgba[s.pedicel] = [0.35, 0.45, 0.15, 1.0]
        m.site_pos[s.stem_end], m.site_pos[s.cut_k], m.site_pos[s.anchor] = stem, cut, top
        # equality data
        if kind == "connect":
            m.eq_data[s.eq] = np.concatenate([top, centre + top, np.zeros(5)])
        else:
            # anchor at the CoM (body2 = world frame); relpose = anchor in the fruit frame + conj(fruit quat)
            m.eq_data[s.eq] = np.concatenate([centre, np.zeros(3), _quat_conj(quat), [1.0]])
        # teleport, still, hold
        d.qpos[s.qadr:s.qadr + 3] = centre
        d.qpos[s.qadr + 3:s.qadr + 7] = quat
        d.qvel[s.dadr:s.dadr + 6] = 0.0
        d.qacc_warmstart[s.dadr:s.dadr + 6] = 0.0
        d.xfrc_applied[s.body] = 0.0
        d.eq_active[s.eq] = 1
        trees_mod.hide_geom(m, self._static_gid[fruit_id])
        self._st["assign"][s.k] = {"fruit_id": fruit_id, "tree_id": f["tree_id"]}
        if forward:
            mujoco.mj_forward(m, d)
        return s.k

    def assign_tree(self, d: mujoco.MjData, tree_id: str, fruit_ids: Optional[Sequence[str]] = None) -> Dict[str, int]:
        """Assign the given fruit (default: the tree's harvest-stage, in-band, unbagged fruit) -> {fruit_id: slot}.

        All or nothing: every fruit is checked (refusals, duplicates, free slots per kind) before the first write,
        and should an assignment still fail, the ones already made are recycled (statics shown again) before the
        PoolError propagates. ``mj_forward`` runs either way, so derived data match the model."""
        ids = list(fruit_ids) if fruit_ids is not None else [
            fid for fid, f in self.fruit.items() if f["tree_id"] == tree_id and f["harvest_stage"]
            and f["in_band"] and not f["bagged"] and f["pool_kind"] in ("connect", "weld")]
        if len(set(ids)) != len(ids):
            raise PoolError(f"fruit listed twice: {sorted({i for i in ids if ids.count(i) > 1})}")
        need: Dict[str, int] = {}
        for fid in ids:
            kind = self._check_assignable(fid)
            need[kind] = need.get(kind, 0) + 1
        for kind, n in need.items():
            free = len(self.free_slots(kind))
            if n > free:
                raise PoolError(f"no free {kind} slot for {n} fruit ({free} free, 5 per kind; recycle first)")
        out: Dict[str, int] = {}
        try:
            for fid in ids:
                out[fid] = self.assign(d, fid, forward=False)
        except Exception:
            for k in out.values():
                self.recycle(d, k, harvested=False, forward=False)
            raise
        finally:
            mujoco.mj_forward(self.m, d)
        return out

    def release(self, d: mujoco.MjData, slot: int) -> None:
        """Equality off (logical cut / twist release): the fruit becomes a free body."""
        if slot not in self._st["assign"]:
            raise PoolError(f"slot {slot} is not assigned")
        d.eq_active[self.slots[slot].eq] = 0

    def is_attached(self, d: mujoco.MjData, slot: int) -> bool:
        return bool(d.eq_active[self.slots[slot].eq])

    def recycle(self, d: mujoco.MjData, slot: int, harvested: bool = False, forward: bool = True) -> None:
        """Park the slot and restore its template; un-hide the static geom unless ``harvested``."""
        s = self.slots[slot]
        a = self._st["assign"].pop(slot, None)
        pool_mod.unpack_slot(self.m, s, self._park[slot])
        pos = pool_mod.park_pos(slot)
        d.qpos[s.qadr:s.qadr + 3] = pos
        d.qpos[s.qadr + 3:s.qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[s.dadr:s.dadr + 6] = 0.0
        d.qacc_warmstart[s.dadr:s.dadr + 6] = 0.0
        d.xfrc_applied[s.body] = 0.0
        d.eq_active[s.eq] = 1
        if a:
            if harvested:
                self._st["harvested"].add(a["fruit_id"])
            else:
                trees_mod.unhide_geom(self.m, self._static_gid[a["fruit_id"]])
        if forward:
            mujoco.mj_forward(self.m, d)

    def reset(self, d: mujoco.MjData, forward: bool = True) -> None:
        for k in range(pool_mod.N_SLOTS):
            self.recycle(d, k, harvested=False, forward=False)
        for fid in list(self._st["harvested"]):
            trees_mod.unhide_geom(self.m, self._static_gid[fid])
        self._st["harvested"].clear()
        if forward:
            mujoco.mj_forward(self.m, d)

    # -------------------------------------------------------------- measurements
    def _site(self, d, sid) -> np.ndarray:
        return np.array(d.site_xpos[sid])

    def cut_point(self, d, slot: int) -> np.ndarray:
        return self._site(d, self.slots[slot].cut_k)

    def stem_end(self, d, slot: int) -> np.ndarray:
        return self._site(d, self.slots[slot].stem_end)

    def anchor_world(self, slot: int) -> np.ndarray:
        """Where the equality holds the fruit in the world (connect: pedicel top; weld: CoM)."""
        s = self.slots[slot]
        e = self.m.eq_data[s.eq]
        return np.array(e[3:6] if s.kind == "connect" else e[0:3])

    def anchor_error_m(self, d, slot: int) -> float:
        """|fruit-side anchor point - world anchor| (connect: pedicel-top site; weld: CoM)."""
        s = self.slots[slot]
        p = self._site(d, s.anchor) if s.kind == "connect" else np.array(d.xpos[s.body])
        return float(np.linalg.norm(p - self.anchor_world(slot)))

    def pedicel_error_mm(self, d, slot: int, cut_world, setpoint_m: Optional[float] = None) -> float:
        """|distance(cut point, stem end) - setpoint| in mm, measured in the fruit frame (OS 3.4)."""
        s = self.slots[slot]
        f = self.fruit[self._st["assign"][slot]["fruit_id"]]
        sp = f["pedicel_leave_m"] if setpoint_m is None else float(setpoint_m)
        R = np.array(d.xmat[s.body]).reshape(3, 3)
        local = R.T @ (np.asarray(cut_world, float) - np.array(d.xpos[s.body]))
        stem_local = np.array(self.m.site_pos[s.stem_end])
        return 1000.0 * abs(float(np.linalg.norm(local - stem_local)) - sp)

    def reaction(self, d, slot: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """(force, torque) the equality applies to the fruit, world frame (torque None for connect).

        Read from ``efc_force`` of the equality rows after ``mj_forward`` / ``mj_step``. For the weld (torquescale 1)
        rows 3-5 are rotational; MuJoCo's quaternion-error Jacobian is half-angle, so the torque is 0.5 x those rows
        (verified against an applied ``xfrc_applied`` torque in tests/test_fruit_pool.py)."""
        s = self.slots[slot]
        rows = np.where((d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == s.eq))[0]
        if rows.size == 0:
            return np.zeros(3), (None if s.kind == "connect" else np.zeros(3))
        f = np.array(d.efc_force[rows])
        if s.kind == "connect":
            return f[0:3], None
        return f[0:3], 0.5 * f[3:6]

    def stem_axis(self, d, slot: int) -> np.ndarray:
        s = self.slots[slot]
        R = np.array(d.xmat[s.body]).reshape(3, 3)
        f = self.fruit[self._st["assign"][slot]["fruit_id"]]
        return R @ np.asarray(f["stem_axis"], float)

    def stem_axis_torque(self, d, slot: int) -> float:
        """The equality's reaction torque on the fruit about its stem axis (N.m; weld slots only)."""
        _, tau = self.reaction(d, slot)
        if tau is None:
            raise PoolError("connect slots carry no torque (a ball joint)")
        return float(np.dot(tau, self.stem_axis(d, slot)))

    def angular_rate_dps(self, d, slot: int) -> float:
        s = self.slots[slot]
        return math.degrees(float(np.linalg.norm(d.qvel[s.dadr + 3:s.dadr + 6])))

    def bump(self, d, slot: int, force_n: Sequence[float], duration_s: float, steps_cb=None) -> None:
        """Push the fruit with a constant world-frame force for ``duration_s`` (a brush by the robot)."""
        s = self.slots[slot]
        n = max(1, int(round(duration_s / self.m.opt.timestep)))
        d.xfrc_applied[s.body, 0:3] = force_n
        for _ in range(n):
            mujoco.mj_step(self.m, d)
            if steps_cb:
                steps_cb(d)
        d.xfrc_applied[s.body, 0:3] = 0.0


def settle_time_after_bump(pool: FruitPool, d, slot: int, impulse_ns: float = 0.05, duration_s: float = 0.02,
                           horizon_s: float = 2.0, threshold_dps: float = 5.0) -> dict:
    """Bump a hanging fruit sideways and measure when its angular rate stays below ``threshold_dps``."""
    m = pool.m
    f = impulse_ns / duration_s
    pool.bump(d, slot, [f, 0.0, 0.0], duration_s)
    dt = m.opt.timestep
    rates, peak, last_above = [], 0.0, 0.0
    cut0 = pool.cut_point(d, slot)
    max_cut_move, max_anchor_err = 0.0, 0.0
    steps = int(round(horizon_s / dt))
    for i in range(steps):
        mujoco.mj_step(m, d)
        r = pool.angular_rate_dps(d, slot)
        rates.append(r)
        peak = max(peak, r)
        if r >= threshold_dps:
            last_above = (i + 1) * dt
        max_cut_move = max(max_cut_move, float(np.linalg.norm(pool.cut_point(d, slot) - cut0)))
        max_anchor_err = max(max_anchor_err, pool.anchor_error_m(d, slot))
    return {"peak_dps": peak, "settle_s": last_above, "rate_at_1s_dps": rates[min(len(rates) - 1, int(1.0 / dt))],
            "max_cut_k_move_m": max_cut_move, "max_anchor_err_m": max_anchor_err}


# ---------------------------------------------------------------------- twist release statistic (reference)
@dataclass
class TwistReleaseStat:
    """OS 3.4 release rule: stem-axis reaction minus the rest baseline (captured at the first update after
    assignment), low-passed with ``lpf_s``, above ``tau_detach`` for ``ticks`` consecutive updates, AND an
    accumulated relative fruit roll >= ``min_roll_deg``. Reference implementation for the WP23 sweep; WP2
    ``twist_logic`` owns the runtime rule."""
    tau_detach: float
    dt: float
    lpf_s: float = 0.040
    ticks: int = 3
    min_roll_deg: float = 30.0
    baseline: Optional[float] = None
    filtered: float = 0.0
    above: int = 0
    peak: float = 0.0

    def update(self, stem_torque: float, roll_deg: float) -> bool:
        if self.baseline is None:
            self.baseline = float(stem_torque)
        x = abs(float(stem_torque) - self.baseline)
        a = self.dt / (self.lpf_s + self.dt)
        self.filtered += a * (x - self.filtered)
        self.peak = max(self.peak, self.filtered)
        self.above = self.above + 1 if self.filtered > self.tau_detach else 0
        return self.above >= self.ticks and abs(roll_deg) >= self.min_roll_deg


def _orchard_lite_model():
    os.environ["NANDI_WORLD"] = "orchard_lite"
    from atharv_nandi.core import farm_scene
    return farm_scene.load_farm_model()


def tau_sweep(taus: Sequence[float] = (0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0), ramp_nm_s: float = 0.1,
              hold_s: float = 10.0, bump_n: float = 2.0, bump_s: float = 0.05, m=None) -> dict:
    """Sweep ``tau_detach`` over weld fruit (one in-band guava and the in-band papaya of orchard_lite).

    Per fruit, the stem-axis statistic (|reaction - baseline|, 40 ms low-pass) is recorded in three scenarios:
    (hold) 10 s hanging untouched, (bump) a ``bump_n`` tangential brush at the fruit side surface for ``bump_s`` (force through the CoM plus its moment about the stem axis), (twist) an applied torque about
    the stem axis ramping at ``ramp_nm_s`` (the treesim probe ramp). For each tau the result says whether hold
    or bump would have crossed it (a false release, ignoring the roll gate) and the twist time to fire. The roll
    gate (>= 30 deg) is the tool's measured roll in WP2; the weld holds the fruit, so it is reported, not used."""
    m = m if m is not None else _orchard_lite_model()
    pool = FruitPool(m)
    dt = float(m.opt.timestep)
    picks = []
    for crop in ("guava", "papaya"):
        f = next(f for f in pool.fruit.values() if f["crop"] == crop and f["in_band"] and f["harvest_stage"])
        picks.append(f)
    out = {"dt": dt, "ramp_nm_s": ramp_nm_s, "hold_s": hold_s, "bump_n": bump_n, "bump_s": bump_s, "fruit": []}
    for f in picks:
        rec = {"fruit_id": f["fruit_id"], "crop": f["crop"], "mass_kg": f["mass_kg"]}
        for scen in ("hold", "bump", "twist"):
            d = mujoco.MjData(m)
            mujoco.mj_resetDataKeyframe(m, d, 0)
            pool.reset(d)
            slot = pool.assign(d, f["fruit_id"])
            for _ in range(int(0.5 / dt)):
                mujoco.mj_step(m, d)
            stat = TwistReleaseStat(tau_detach=1e9, dt=dt)
            stat.update(pool.stem_axis_torque(d, slot), 0.0)
            series = []
            s = pool.slots[slot]
            q0 = np.array(d.qpos[s.qadr + 3:s.qadr + 7])
            max_rot = 0.0
            T = hold_s if scen == "hold" else (1.5 if scen == "bump" else 12.0)
            for i in range(int(T / dt)):
                t = i * dt
                if scen == "bump":
                    # tangential brush at the fruit's side surface: force F along x applied at r = (0, rx, 0)
                    # from the CoM, i.e. F through the CoM plus the moment r x F (about the stem axis)
                    on = 0.2 <= t < 0.2 + bump_s
                    rx = float(m.geom_size[s.fruit][1])
                    F = np.array([bump_n, 0.0, 0.0]) if on else np.zeros(3)
                    d.xfrc_applied[s.body, 0:3] = F
                    d.xfrc_applied[s.body, 3:6] = np.cross([0.0, rx, 0.0], F)
                elif scen == "twist":
                    d.xfrc_applied[s.body, 3:6] = ramp_nm_s * t * pool.stem_axis(d, slot)
                mujoco.mj_step(m, d)
                stat.update(pool.stem_axis_torque(d, slot), 0.0)
                series.append(stat.filtered)
                q = np.array(d.qpos[s.qadr + 3:s.qadr + 7])
                max_rot = max(max_rot, math.degrees(2.0 * math.acos(min(1.0, abs(float(np.dot(q, q0)))))))
            series = np.asarray(series)
            rec[f"{scen}_max_fruit_rotation_deg"] = round(max_rot, 4)
            if scen == "twist":
                fire = {}
                for tau in taus:
                    above = series > tau
                    run, t_fire = 0, None
                    for i, a in enumerate(above):
                        run = run + 1 if a else 0
                        if run >= 3:
                            t_fire = (i + 1) * dt
                            break
                    fire[f"{tau:g}"] = None if t_fire is None else round(t_fire, 3)
                rec["twist_fire_s"] = fire
                rec["twist_lag_s"] = {k: (None if v is None else round(v - float(k) / ramp_nm_s, 3))
                                      for k, v in fire.items()}
            else:
                rec[f"{scen}_peak_nm"] = round(float(series.max()), 5)
            pool.reset(d)
        peak = max(rec["hold_peak_nm"], rec["bump_peak_nm"])
        rec["false_release"] = {f"{tau:g}": bool(peak > tau) for tau in taus}
        ok_f = [t for t in taus if t >= 2.0 * peak]
        rec["recommended_tau_nm"] = min(ok_f) if ok_f else None
        out["fruit"].append(rec)
    worst = max(max(r["hold_peak_nm"], r["bump_peak_nm"]) for r in out["fruit"])
    ok = [t for t in taus if t >= 2.0 * worst]
    out["worst_false_peak_nm"] = round(worst, 5)
    out["recommended_tau_nm"] = min(ok) if ok else None
    out["rule"] = "smallest swept tau >= 2 x the worst hold/bump peak (2x margin; assumption)"
    return out


def main(argv: Sequence[str]) -> int:
    if "--sweep" in argv:
        print(json.dumps(tau_sweep(), indent=1))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
