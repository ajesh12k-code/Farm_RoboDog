"""WPCAP compose module ``contact_pairs`` (order 60): per-pair contact parameters for grasp and transfer.

Orchard worlds only (it attaches when the fruit pool exists, like ``fruit_pool``), so the legacy world and every
legacy KPI stay byte-identical. Three pair families (docs/cap-libraries.md section 3 has the table and the
measured numbers):

``finger_fruit``  explicit MuJoCo ``<pair>`` for each finger pad x each pool fruit geom. Explicit pairs replace the
                  dynamic contact of the same geom pair (MuJoCo skips a dynamic pair that is also explicit), so the
                  grip uses exactly these numbers instead of the pad's priority-1 geom params: condim 4 (adds
                  torsional friction, so a fruit does not spin inside the grip during a twist or a fast yaw),
                  tangential friction 1.2, torsional 0.005 m, the pads' stiff solref/solimp (creep < 1 mm/s).
                  A parked slot is at the park spot (-20, -20, 1), far from the pads, so the pair never fires
                  for it even though explicit pairs ignore contype / conaffinity.
``fruit_tray``    the desap tray cells, lane walls, divider, papaya cradle pads and the basket floor get contact
                  ``priority`` 2 with the tray parameters, so every fruit-tray contact uses them whatever the
                  fruit's own params (priority is per geom; the tray only ever touches produce: its contype is 0
                  and only produce with WP3's tray bit or contype 1 on the basket floor reaches it). condim 6 with
                  rolling friction stops a placed fruit rocking or rolling in its cell; solref 0.01 (2.5 x the
                  pads') damps the landing so a fruit does not bounce.
``foot_soil``     **not changed** (the physics alignment, stability matrix and gait numbers are measured on it).
                  ``foot_soil_effective(model)`` reports what MuJoCo's geom mixing gives a foot on the orchard
                  ground, and tests/test_grasp_contacts.py pins it and checks it against a live contact: condim 3,
                  friction 0.8 / 0.02 / 0.01, solref 0.01 1 (the hard-ground override farm_scene puts on every
                  terrain geom); standing penetration and stance-foot slip are measured there too.
"""
import fnmatch
from typing import Dict, List

import mujoco

from atharv_nandi.sim.world import compose

VERSION = "1"
ORDER = 60
PAD_GEOMS = ("finger_left_pad", "finger_right_pad")
FRUIT_GEOM_PATTERN = "pool_*_fruit"
TRAY_GEOMS = ("basket_floor", "basket_divider", "tray_lane_wall_l", "tray_lane_wall_r", "tray_div_l0", "tray_div_l1",
              "tray_div_l2", "tray_div_r0", "tray_div_r1", "tray_div_r2", "papaya_cradle_l", "papaya_cradle_r")
FOOT_GEOMS = tuple(f"{leg}_foot" for leg in ("FR", "FL", "MR", "ML", "RR", "RL"))

PAIRS: Dict[str, dict] = {
    "finger_fruit": {"condim": 4, "friction": [1.2, 1.2, 0.005, 0.0001, 0.0001], "solref": [0.005, 1.0],
                     "solimp": [0.995, 0.999, 0.001, 0.5, 2.0], "margin": 0.0, "gap": 0.0},
    "fruit_tray": {"priority": 2, "condim": 6, "friction": [1.0, 0.05, 0.02], "solref": [0.01, 1.0],
                   "solimp": [0.95, 0.99, 0.001, 0.5, 2.0]},
}
# What the unchanged foot-soil mixing gives (pinned by tests/test_grasp_contacts.py).
FOOT_SOIL_EXPECTED = {"condim": 3, "friction": [0.8, 0.8, 0.02, 0.01, 0.01], "solref": [0.01, 1.0]}


def _geom_names(spec) -> List[str]:
    return [g.name for g in spec.geoms if g.name]


def apply(spec, pad_geoms=PAD_GEOMS, fruit_pattern: str = FRUIT_GEOM_PATTERN, tray_geoms=TRAY_GEOMS) -> dict:
    """Add the finger-fruit pairs and the tray priority params to ``spec``; returns what was applied."""
    names = set(_geom_names(spec))
    fruits = sorted(n for n in names if fnmatch.fnmatch(n, fruit_pattern))
    pads = [p for p in pad_geoms if p in names]
    ff = PAIRS["finger_fruit"]
    for p in pads:
        for f in fruits:
            spec.add_pair(name=f"cp_{p}_{f}", geomname1=p, geomname2=f, condim=ff["condim"],
                          friction=list(ff["friction"]), solref=list(ff["solref"]), solimp=list(ff["solimp"]),
                          margin=ff["margin"], gap=ff["gap"])
    ft = PAIRS["fruit_tray"]
    trays = []
    for g in spec.geoms:
        if g.name in tray_geoms:
            g.priority = ft["priority"]
            g.condim = ft["condim"]
            g.friction = list(ft["friction"])
            g.solref = list(ft["solref"])
            g.solimp = list(ft["solimp"])
            trays.append(g.name)
    return {"pairs": len(pads) * len(fruits), "pads": pads, "fruits": fruits, "tray_geoms": trays}


def _attach(spec, field, ctx):
    if not ctx.config.get("tree_plans"):
        return          # legacy world: unchanged
    apply(spec)


def _mix_solref(m: mujoco.MjModel, g1: int, g2: int) -> List[float]:
    if m.geom_priority[g1] != m.geom_priority[g2]:
        g = g1 if m.geom_priority[g1] > m.geom_priority[g2] else g2
        return [float(v) for v in m.geom_solref[g]]
    s1, s2 = float(m.geom_solmix[g1]), float(m.geom_solmix[g2])
    mix = 0.5 if s1 + s2 < mujoco.mjMINVAL else s1 / (s1 + s2)
    return [float(mix * a + (1 - mix) * b) for a, b in zip(m.geom_solref[g1], m.geom_solref[g2])]


def mixed_params(m: mujoco.MjModel, g1: int, g2: int) -> dict:
    """Contact params MuJoCo's geom mixing gives a dynamic contact of g1 and g2 (priority, max friction, solmix)."""
    if m.geom_priority[g1] != m.geom_priority[g2]:
        g = g1 if m.geom_priority[g1] > m.geom_priority[g2] else g2
        fr = [float(v) for v in m.geom_friction[g]]
        condim = int(m.geom_condim[g])
    else:
        fr = [float(max(a, b)) for a, b in zip(m.geom_friction[g1], m.geom_friction[g2])]
        condim = int(max(m.geom_condim[g1], m.geom_condim[g2]))
    return {"condim": condim, "friction": [fr[0], fr[0], fr[1], fr[2], fr[2]], "solref": _mix_solref(m, g1, g2)}


def foot_soil_effective(m: mujoco.MjModel, ground_geom: str = "orchard_ground") -> dict:
    """Mixed foot-ground params for every foot (they are identical; returns the first plus a per-foot check)."""
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, ground_geom)
    if gid < 0:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    per = {f: mixed_params(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f), gid) for f in FOOT_GEOMS}
    first = per[FOOT_GEOMS[0]]
    return {**first, "uniform": all(v == first for v in per.values()),
            "ground": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)}


compose.register_module("contact_pairs", ORDER, _attach, version=VERSION,
                        fingerprint=lambda field, ctx: f"v{VERSION},ff={PAIRS['finger_fruit']['condim']}"
                        f"/{PAIRS['finger_fruit']['friction'][0]:g},tray={PAIRS['fruit_tray']['solref'][0]:g}")
