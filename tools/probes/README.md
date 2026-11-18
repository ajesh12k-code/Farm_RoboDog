# Feasibility probes (scratch evidence, not CI)

These scripts produced every number marked **(measured, probe)** in the orchard MVP design spec and research notes. They ran on 2026-11-17 against `feat/nandi-pilot` @ 5e7bf95 in the `atharv-nandi-sim` image (MuJoCo 3.3.7) with 2 CPU / 2 GB and a 180 s cap, on a Mac. They carry no run_id or device_id, so they are **not KPIs** (run-identity rule). WP0c and WP11a re-run them on a CI runner and replace each number with a run-id-tagged value.

They are kept as written (scratch quality, not covered by tests or coverage gates). Run from the repo root unless noted.

| Script | What it measures | How it was run |
|---|---|---|
| `reach_proto.py` | crown-mast prototype: static CoM / 6-foot and tripod margins vs extension, slope (rotated gravity), head mass and mast x; wind via `xfrc_applied`; tripod walk tilt; live body_pos ramp; creep | `python3 tools/probes/reach_proto.py <mode>` with modes `static`, `wind`, `sens`, `lift`, `walk`, `ramp` (lumped single stage, mast at x −0.36 unless set) |
| `reach_extra.py` | extra points: 5° cross-slope, walk at 0.3 cmd, 20 s standing creep at 0° and 10° | imports `reach_proto` from the same directory |
| `treesim_proto.py` | procedural trees via MjSpec; physics ms/step; detach (mango connect pull, guava weld twist); render ms/pass | `python3 tools/probes/treesim_proto.py phys|detach|render2` with `NANDI_TERRAIN=flat`, `LP_NUM_THREADS=1`, Xvfb + glfw for render |
| `cull_test.py` | alpha-0 cull effect on scene geoms and render time (superseded by the mjvScene post-filter in ORCHARD-MVP-SPEC 3.7) | run next to `treesim_proto.py` |
| `orbit_probe.py`, `sweep.sh`, `sweep2.sh` | side-mode and face-mode orbit feasibility: radius, drift, speed, bearing error, roll/pitch, yaw jitter | `python3 tools/probes/orbit_probe.py side|face <cmd_lin> <cmd_yaw> <speed_scale> <sim_s> [terrain]`; the repo root on `PYTHONPATH` (the image sets `/ws`); `NANDI_CROPS=1`, `NANDI_SEED=7` |
| `calc.py` | arithmetic: elevation to canopy top, mm/px, orbit times, storage per tree | plain Python |

Known limits: the mast probe used one lumped moving stage (the spec now has three nested sections), and no rack, tools, basket or arm load; rough and ridges terrain rows of the orbit probe were identical (weak evidence).

## WP0c re-run (2026-11-17)

`tools/feasibility_sweep.py` replaces the reach, load and stability numbers of `reach_proto.py` / `reach_extra.py`. It uses the physics-aligned model and the full sectioned layout: mast at x −0.30, rack and tools, 3 kg basket, T4 + fruit. Results and run ids are in `config/fields/orchard_candidates.json`; the write-up is feasibility-report. The probes above stay as historical evidence for the pre-alignment numbers in the spec. They are not re-run, because the model they patch (`atharv_nandi.xml` at 5e7bf95, kp 120 legs, pyramidal cones) no longer exists at HEAD.
