# WPCAP: arm motion planning, hardware abstraction and contact/grasp physics

Status: everything here is [BUILT] on `feat/nandi-wpcap` (code wins). This document is the package contract for WPCAP; [hld.md](hld.md) indexes it. Related: [hal-sim-to-real.md](hal-sim-to-real.md) (HAL options A and B), [sim-fidelity-register.md](sim-fidelity-register.md) (servo and gripper rows), [state-machines.md](state-machines.md) 3.1 (the grasp false positive).

Files: `atharv_nandi/core/planning/{__init__,collision_scene,ompl_planner}.py`, `atharv_nandi/core/hal/{__init__,interfaces,sim_backend,stub_backend,controllers}.py`, `atharv_nandi/sim/world/modules/contact_pairs.py`, `config/hal.yaml`, `docker/Dockerfile` (`ompl==2.0.1`, `pytest-cov`), `requirements.txt`, `requirements-ci.txt`, `tests/test_ompl_planner.py`, `tests/test_hal.py`, `tests/test_grasp_contacts.py`, this file. No topic, service, joint, actuator or equality is added. nq, nv and nu are unchanged.

All numbers below are MEASURED on 2026-11-17 in the capped sim runner (Docker on the Mac, 2 CPU / 2 GB, image `atharv-nandi-sim:wpcap`, MuJoCo 3.3.7, Python 3.10.12), unless they are marked otherwise.

## 1 Arm motion planning (`controllers.planning`)

### 1.1 Backends

| Backend | What | When |
|---|---|---|
| `ompl` | OMPL 2.0.1 Python bindings (PyPI wheel, nanobind). RRT-Connect in a 5-D `RealVectorStateSpace` bounded by the arm joint limits. OMPL's simplifier is not used, because its B-spline pass adds states and checks | `import ompl` works (sim image) |
| `rrtconnect` | Pure-Python/NumPy RRT-Connect (Kuffner and LaValle) with the same validity checker. Seeded, so it is reproducible | documented fallback (CI venv without the wheel) |
| `auto` (default) | `ompl` if it is importable, else `rrtconnect` | |

Wheel check: PyPI `ompl` 2.0.1 publishes `manylinux_2_17` aarch64 and x86_64 wheels for cp310 to cp313, plus macOS wheels, and needs no system libraries. It installs in the sim image (py3.10 aarch64). `pip download --only-binary=:all: ompl==2.0.1` on `python:3.11-slim` aarch64 fetches `ompl-2.0.1-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`.

Both backends then share these steps:
1. Greedy and random shortcutting.
2. An independent dense re-check of the final path (`verify_step` 0.005, max-norm, rad or m). If it fails because the coarse 0.02 edge checks skipped a thin obstacle, the planner re-plans once with edges at `verify_step`.
3. Stop-at-waypoint trapezoidal time parameterization, under the per-joint `max_velocity` / `max_acceleration` from `config/hal.yaml`.

### 1.2 API (for WP8)

```python
from controllers.planning import ArmPlanner
planner = ArmPlanner.from_config(model)               # once per process; config/hal.yaml `planning`
res = planner.plan(None, q_goal, data=data, model=model,  # start = arm joints in data
                   ignore=["pool_3"],                   # the target fruit body (and its pedicel)
                   attach=[],                           # bodies carried with the wrist (held fruit / tool)
                   ignore_classes=())
res.ok, res.reason        # "" | start_invalid | goal_invalid | no_solution | verify_failed
res.path                  # N x 5 (gantry, yaw, shoulder, elbow, wrist_pitch), path[0] = start, path[-1] = goal
res.trajectory.sample(t)  # Trajectory: t knots, duration; .samples(dt)
res.min_clearance         # per obstacle class, m (capped 0.10)
res.contacts              # the violations when start/goal is refused
```

`CollisionScene` can be used on its own: `is_valid(q5)`, `contacts(q5)`, `clearance(q5)`, `motion_valid(qa, qb, step)`, `tip(q5)`, `attach(body)`, `set_ignore(...)`, `sync(model, data)`. It works on a `deepcopy` of the live model, so the live sim is never touched. A test pins this.

### 1.3 Collision model

The moving set is every collidable geom of `arm_turret` and its children, plus any attached bodies. `attach(body)` takes a free root and carries its whole subtree: a mounted tool's head, blades and jaws hang on their own hinge / slide joints under the handle and ride with the wrist at their current joint values (WP8 change request; `test_attach_carries_the_whole_subtree`). WP8's `subtree_attach` wrapper in `sim_executor.py` is now redundant and harmless; WP8 can delete it. The obstacles are every geom that can physically touch the moving set (the contype/conaffinity rule of the live model), plus keep-out robot parts that are contype 0 in the sim but solid on hardware. In the private copy, moving geoms get `contype = 1<<7` and obstacles get `conaffinity = 1<<7`. Body masks are recomputed too, because MuJoCo filters body pairs first. As a result `mj_collision` only produces arm-versus-obstacle pairs. A class's clearance is its geom margin. Explicit `<pair>`s (the 20 finger x pool-fruit pairs of section 3) bypass contype/conaffinity, so the scene gives a pair the obstacle class margin when it joins a moving geom to an obstacle, and counts only moving x obstacle contacts: a pad touching the held (attached) fruit, an ignored target, or a parked contype-0 slot never makes a state invalid. A test plans a carry with the pads closed 4 mm into a held fruit. One check is `mj_kinematics` plus `mj_collision`, about 40-60 µs in orchard_demo (1,161 geoms, 185 obstacles).

| Class | Members | Margin (m) |
|---|---|---|
| `robot` | trunk, rail, basket, legs, rack, bumper | 0.005 (WP0c `ROBOT_CLEAR_M`) |
| `carriage` | gantry carriage vs the upper arm | 0.000 (actual contact, as `DorsalArmIKSolver.robot_collisions`) |
| `keepout` | `mast_*`, `crown_head*`, `tray_*`, `cradle_*` (contype 0 in the sim) | 0.010 |
| `wood` | `tree_*` collidable geoms (keep wood, contype 1) | 0.010 |
| `fruit` | assigned pool fruit and legacy crops | 0.010 |
| `fixture` | crate, dock, props | 0.010 |
| `ground` | floor / orchard terrain | 0.020 (WP0c `GROUND_CLEAR_M`) |

Run-time mutations are picked up by `sync()` before every plan: pool assign or release (geom and body contype, size, pose) and mast `body_pos`.

### 1.4 Results

| Scene | Backend | Result |
|---|---|---|
| All 99 FROZEN WP0c targets (mango cut, guava clip/cradle, papaya cradle, skirt prune) from home, robot at its station, no extra obstacle | ompl | 99/99 planned; 92 are direct (2 waypoints); mean 0.051 s, max 0.407 s |
| 69 pick targets (mango, guava, papaya); a 6 cm "branch" (wood class) on the straight joint path, so a detour is needed | ompl | 69/69; p50 0.145 s, p95 0.436 s, max 0.483 s; 9.7 waypoints mean |
| same | rrtconnect | 69/69; p50 0.133 s, p95 0.189 s, max 0.363 s |
| unit test: 17 targets (one per tree × pick task), blocked direct path | ompl / rrtconnect | 17/17 each; max 0.506 / 0.521 s; min wood clearance 0.0100 / 0.0101 m |
| `slow` test: 34 targets (two per tree × task) | ompl / rrtconnect | 34/34 each; max 0.580 / 0.480 s |

Every path in the tests is re-checked by a second scene with all margins at 0 (physical contact) at 0.0025 steps. Acceptance "≤ 1 s per typical pick plan" holds with 0.42 s headroom at the worst case seen. OMPL is not faster than the fallback here, because the cost is the Python validity callback, not the search.

## 2 Hardware abstraction (`controllers.hal`)

This is option B of [hal-sim-to-real.md](hal-sim-to-real.md) (Python, unit-testable) with the names and semantics of option A (`ros2_control`), so moving to the C++ stack later changes no consumer.

| ros2_control | here |
|---|---|
| `<ros2_control>` tag | `config/hal.yaml` (`schema: nandi.hal/1`): hardware blocks `nandi_arm` (5 arm joints + 2 fingers + `tool_flange_ft` sensor) and `nandi_legs` (18 leg joints + `imu` sensor), `plugins` per backend |
| `SystemInterface` | lifecycle callbacks `on_init/configure/activate/deactivate/cleanup/shutdown/error`, `export_*_interfaces`, `read`, `write`, `prepare_/perform_command_mode_switch` |
| interfaces | `<joint>/position`, `/velocity`, `/effort` (command and state are separate values); sensor states such as `imu/orientation.w` and `tool_flange_ft/force.z` |
| `ResourceManager` | lifecycle `unconfigured → inactive → active → finalized`; **exclusive, all-or-none claims**; releasing a claim (controller deactivate) zeroes a velocity / effort command and switches the joint back to a position hold at the current pose (servo restored), so nothing keeps moving after its controller stops; one command interface per joint at a time; commands clamped to the interface min/max (joint limits); a read or write ERROR runs `on_error`, drops the component to `unconfigured`, releases its claims and deactivates the controllers that held them |
| controllers | `JointStateBroadcaster` (dict shaped like `sensor_msgs/JointState`), `ForwardCommandController`, `JointTrajectoryController` (executes a planner `Trajectory`), `ControllerManager.update` = read → update → write |

Backend selection: `backend:` in `config/hal.yaml`, overridden by env `NANDI_HAL_BACKEND` (`mujoco` | `stub`). Env `NANDI_HAL_CONFIG` points to another file.

| Backend | position | velocity | effort | states |
|---|---|---|---|---|
| `MujocoSystem` (`sim_backend.py`) | `ctrl = cmd` + optional gravity feed-forward `qfrc_bias/kp` (`gravity_feedforward: true` on the arm joints) | the target integrates `v·dt`, clamped to ctrlrange | that joint's servo gain and bias are set to 0 on the live model (restored on switch-back or deactivate); `qfrc_applied = cmd` clamped to forcerange | qpos, qvel, `qfrc_actuator + qfrc_applied`, named MuJoCo sensors |
| `StubSystem` (`stub_backend.py`) | first-order lag τ 0.03 s with `max_velocity` | first-order lag to the clamped velocity | damped inertia | integrated state; optional delay (`delay_steps`) and seeded noise (`noise_std`); faults `comm_timeout` (read) and `overcurrent:<joint>` (write) |

`read`/`write` never step physics. The sim driver (WP0b) or the caller steps between them.

Measured: on MuJoCo (legacy world, 100 Hz controller, 500 Hz physics), a planned 5-joint arm move of 1.26 s has a max tracking error of 0.042 rad·m (max-norm) and a final error below 1e-4, and the robot stays standing. On the stub it has a max tracking error of 0.023. Velocity mode: 0.5 rad/s for 1 s gives 0.5 ± 0.05 rad. Effort mode: the 9 N·m request is clamped to 5 N·m, and the servo gains are restored byte-identical. After activation the robot holds its pose for 1 s (trunk z > 0.25 m).

## 3 Contact / grasp physics (`atharv_nandi/sim/world/modules/contact_pairs.py`, order 60)

This applies to orchard worlds only (it attaches when the fruit pool exists). The legacy world is byte-identical (`npair` 0, tray priority 0). This is pinned by a test. The world_id fingerprint changes for orchard worlds.

| Pair | How | Params | Why |
|---|---|---|---|
| finger-fruit | explicit `<pair>` per pad × pool fruit (20). It **replaces** the dynamic contact: one contact per pad, measured | condim 4, friction 1.2 / 1.2 / 0.005 (torsional, m) / 1e-4 / 1e-4, solref 0.005 1, solimp 0.995 0.999 0.001 | torsional friction: a fruit gripped off the pad line no longer swings in the jaws |
| fruit-tray | `priority 2` + params on the tray cells, lane walls, divider, papaya cradle pads and the basket floor (these only touch produce) | condim 6, friction 1.0 / 0.05 / 0.02, solref 0.01 1, solimp 0.95 0.99 0.001 | rolling friction stops a placed fruit rocking; softer landing |
| foot-soil | **unchanged** (physics alignment, stability matrix and gait are measured on it); pinned by `FOOT_SOIL_EXPECTED` and checked against live contacts | condim 3, friction 0.8 / 0.02 / 0.01, solref 0.01 1 | |

Measured (orchard_demo, kinematic setups, see `tests/test_grasp_contacts.py`):

| Check | Tuned | Geom mixing only (module off) | Gate |
|---|---|---|---|
| Lift a gripped fruit 10 cm in 1 s from a WP0c side-pick pose; max slip in the wrist frame / residual creep | 0.25 kg: 0.14 mm / 0.10 mm/s; 0.6 kg: 0.09 mm / 0.07 mm/s; guava 0.25 kg: 0.26 mm / 0.19 mm/s | about the same (home-pose probe: 0.36 mm both) | ≤ 2 mm, ≤ 1 mm/s (harvest check allows 6 mm / 0.3 s) |
| Fruit gripped 1.5 cm off the pad line, wrist pitched 0.6 rad: rotation in the grip | 0.65° | 34.8° | ≤ 3° |
| Fruit dropped 1-2 cm into a tray cell, then 0.5 s settle and 1 s watch: drift / max relative speed | 1.3-2.7 µm / 0.010-0.011 mm/s | 2-5 µm / 0.018-0.020 mm/s | ≤ 50 µm, ≤ 1 mm/s |
| Foot-soil, standing 2 s: max penetration / stance foot slip | 0.69 mm / 0.81 mm (unchanged physics) | same | ≤ 1 mm / ≤ 1 mm |
| Yaw swing 1.5 rad/s with a gripped fruit (extra, not a gate) | slip 3.8 mm, then still | 3.8 mm | (`max_velocity` of yaw is 1.2 rad/s) |

Honest reading: the pads' existing priority-1 stiff contact already kept lift slip well under the gate. What the pair adds is torsional grip (35° → 0.65°). The tray had no jitter before either; the tray parameters make that explicit and halve the residual speed. A fruit resting in a cell touches only the basket floor; the cell walls only matter when the fruit is placed off-centre.

## 4 Image and requirements

- `docker/Dockerfile`: `pip3 install "ompl==2.0.1" "pytest-cov>=5,<7"`. No apt packages: `ros-humble-ros2-control` is not used (option B). The test image used here is `atharv-nandi-sim:wpcap`. Rebuild the live image only through Integrate.
- `requirements-ci.txt` adds `pytest-cov`, and `pyyaml` (which `config/*.yaml` loaders already needed). `ompl` stays optional. Verified on plain `python:3.11-slim` aarch64 without ROS: it installs, and `tests/test_ompl_planner.py`, `tests/test_hal.py` and `tests/test_grasp_contacts.py` give 46 passed, 3 skipped (the OMPL-backend cases).
- `requirements.txt` adds `pyyaml` and `ompl==2.0.1` behind environment markers (Python 3.10 to 3.13 on Linux x86_64 / aarch64 or macOS, where PyPI has a wheel), so `pip install -r requirements.txt` skips it elsewhere and the planner falls back to RRT-Connect. musl has no marker: use `requirements-ci.txt` there.

## 5 Open items

- WP8 owns the harvest integration: approach and exit plans via `ArmPlanner.plan` with `ignore=[target pool body]`, carry plans with `attach=[held body]` and `ignore_classes=("fruit",)` near the tray. The planner does not yet constrain the tool axis (TCP orientation) along the path: goals come from IK, and the path between them is free in joint space.
- Foliage (group 3/5 envelopes) is not an obstacle (contype 0). Only wood is. A soft foliage cost, which the LEAF SCAN clearance would need, is not built.
- The HAL is not yet wired into `sim_runner` / ROS topics (`/nandi/joint_states` M1 of the HAL plan). The interfaces and backends are ready for it.
- Contact parameters are sim-tuned, ASSUMED values. Pad material, wet or waxy skin and real squeeze force (fidelity register "Gripper friction") stay open.
