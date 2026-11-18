#!/usr/bin/env python3
"""Closed-loop auto-harvest trials in bare MuJoCo (ROS-free), printing one line per run.

Uses the live-like SimIO of tests/test_harvest_logic.py (500 Hz physics, 50 Hz qpos samples, the
harvest node's command slew, optional slow-sim RTF). Run inside the sim image from the repo root:
    python3 tools/harvest_trials.py [n_random]
"""
import os
import sys
import time

import mujoco
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from atharv_nandi.core.harvest_logic import select_target  # noqa: E402
from tests.test_harvest_logic import _MODEL, _crop_y, run_closed_loop_harvest, trunk_beside  # noqa: E402


def _home_qpos_with_crop1(x, y):
    d = mujoco.MjData(_MODEL)
    mujoco.mj_resetDataKeyframe(_MODEL, d, 0)
    d.qpos[32:34] = [x, y]
    return d.qpos.copy()


def trial(label, **kw):
    t0 = time.time()
    r = run_closed_loop_harvest(_MODEL, **kw)
    good = r["ok"] and r["in_basket"]
    regrasp = "regrasp" in r["io"].phases
    print(f"{'PASS' if good else 'FAIL'} {label:44s} regrasp={regrasp!s:5} sim={r['data'].time:5.1f}s "
          f"wall={time.time() - t0:4.1f}s  {r['msg']}", flush=True)
    return good


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    rng = np.random.default_rng(int(os.environ.get("TRIAL_SEED", "1")))
    if which in ("all", "crop1"):
        res = []
        for i in range(n):
            dx, dy = rng.uniform(-0.02, 0.02, 2)
            settle = float(rng.uniform(0.6, 1.6))
            rtf = 1.0 if i % 3 else float(rng.uniform(0.5, 1.0))
            res.append(trial(f"crop_item_1 d=({dx:+.3f},{dy:+.3f}) settle={settle:.2f} rtf={rtf:.2f}",
                             crop_shift={1: (dx, dy)}, settle=settle, rtf=rtf))
        print(f"crop_item_1 from home: {sum(res)}/{len(res)} in basket", flush=True)
    if which in ("all", "others"):
        res = []
        for k in (2, 3):
            s = 1.0 if _crop_y(k) > 0 else -1.0
            for off in ((0.62, 0.0), (0.45, 0.25 * s), (0.10, 0.45 * s), (-0.30, 0.42 * s)):
                res.append(trial(f"crop_item_{k} at trunk-frame {off}", target=f"crop_item_{k}",
                                 trunk_xy=trunk_beside(_MODEL, k, off)))
        print(f"crop_item_2/3 teleported: {sum(res)}/{len(res)} in basket", flush=True)
    if which in ("all", "grid"):
        # crop_item_1 moved around the standing robot: every crop the planner accepts must end in the
        # basket; the rest must fail cleanly as "out of arm reach" before the arm moves.
        counts = {"harvested": 0, "clean_out_of_reach": 0, "not_selected": 0, "bad": 0}
        ys = np.round(np.arange(-0.6, 0.65, 0.1), 2)
        print("grid (rows x, cols y " + " ".join(f"{y:+.1f}" for y in ys) + "): H harvested, o clean out-of-reach,"
              " . not selected (under body/legs or > 1.08 m), X bad", flush=True)
        for x in np.round(np.arange(-0.6, 0.95, 0.1), 2):
            row = ""
            for y in ys:
                q = _home_qpos_with_crop1(x, y)
                if select_target(q, ["crop_item_1"])[0] is None:
                    counts["not_selected"] += 1
                    row += " ."
                    continue
                r = run_closed_loop_harvest(_MODEL, crop_shift={1: (x - 0.62, y)})
                if r["ok"] and r["in_basket"]:
                    counts["harvested"] += 1
                    row += " H"
                elif "out of arm reach" in r["msg"] and r["io"].phases == ["approach", "return_rest"]:
                    counts["clean_out_of_reach"] += 1
                    row += " o"
                else:
                    counts["bad"] += 1
                    row += " X"
                    print(f"BAD crop at trunk-frame ({x:+.2f},{y:+.2f}): {r['msg']} phases={r['io'].phases}", flush=True)
            print(f"x={x:+.1f} {row}", flush=True)
        print(f"grid: {counts}", flush=True)
    if which in ("all", "reach"):
        trial("crop_item_1 shifted +0.33 m (out of reach)", crop_shift={1: (0.33, 0.0)})


if __name__ == "__main__":
    main()
