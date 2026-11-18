"""Grasp and presence sensing for the tools (WP2; MVP-SPEC 2.3 verify step, 3.3 assisted grasp, OS 2.4).

Pure functions (numpy only) plus two MuJoCo readers that take ids from ``registry.ToolHandles``:

    tool_present(flange, handle_ref)       the handle sits on the flange (verify step), <= 1 mm
    cradle_occupied(handle_ref, cradle)    a tool sits in its cradle, <= 5 mm (``tool_present`` per cradle)
    jaw_grip(touch_a, touch_b)             both T4 pads press the fruit (> 1 N each)
    holding(...)                           the swap interlock: any payload held (catch weld, jaw grip, picker
                                           pads on a crop)
    contacts_between(m, d, geoms, bodies)  number of active contacts between tool geoms and the given bodies
                                           (acceptance: 0 tool-finger contacts / 1,000 steps)
"""
from typing import Iterable, Sequence

import numpy as np

PRESENT_TOL_MM = 1.0
CRADLE_TOL_MM = 5.0
JAW_TOUCH_N = 1.0


def _dist_mm(a, b) -> float:
    return 1000.0 * float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def tool_present(flange_pos, handle_pos, tol_mm: float = PRESENT_TOL_MM) -> bool:
    return _dist_mm(flange_pos, handle_pos) <= tol_mm


def cradle_occupied(handle_pos, cradle_pos, tol_mm: float = CRADLE_TOL_MM) -> bool:
    return _dist_mm(handle_pos, cradle_pos) <= tol_mm


def jaw_grip(touch_a: float, touch_b: float, min_n: float = JAW_TOUCH_N) -> bool:
    return float(touch_a) > min_n and float(touch_b) > min_n


def holding(catch_active: bool = False, jaw_a_n: float = 0.0, jaw_b_n: float = 0.0, picker_held: bool = False,
            min_n: float = JAW_TOUCH_N) -> bool:
    return bool(catch_active) or jaw_grip(jaw_a_n, jaw_b_n, min_n) or bool(picker_held)


def contacts_between(m, d, geoms: Iterable[int], bodies: Sequence[int]) -> int:
    gs, bs = set(int(g) for g in geoms), set(int(b) for b in bodies)
    n = 0
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
        if (g1 in gs and b2 in bs) or (g2 in gs and b1 in bs):
            n += 1
    return n


def touch(d, adr: int) -> float:
    return float(d.sensordata[adr])
