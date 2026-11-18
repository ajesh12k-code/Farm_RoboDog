# Test plan

Status: draft. Gaps covered: DOC-01 and the test side of KPI-01. WP12 (2026-11-18) updates this plan for the orchard MVP (ORCHARD-MVP-SPEC 6), maps every mandatory orchard test to its file, and folds in the WP11 evidence.

## 1 Levels

| Level | What it tests | Where it runs | Status |
|---|---|---|---|
| L1 Unit | Pure logic in `atharv_nandi/core/`, `atharv_nandi/core/workflow/`, the hooks on a MuJoCo model, and the design docs themselves; no ROS import | `tests/` with pytest in the `atharv-nandi-sim` image (README, Tests) | [BUILT]: 103 test files (102 on `feat/nandi-pilot` @ 1840118 + `tests/test_wp12_docs.py`); unit-tier selection 2,757 passed, 1 failed (a timing gate, section 5), 25 skipped, 6 xfailed, 53 deselected [MEASURED, capped runner, section 5] |
| L2 Station tier | One preset or behaviour closed-loop in MuJoCo with the real gait and hooks (`slow`, ≤ 3 min each): crown walk, gust, TREE 360 orbit, stuck / sink, low-SoC return | same, `-m slow` | [BUILT]; nightly selection; not re-run by WP12 |
| L3 Headless ROS smoke | All nodes in one container, no viewer; scripted `reset_world` then `auto_harvest`; check basket truth on `/nandi/game_status` | throwaway container, no network | [PROPOSED] gate (G3); `tools/mission_ros_smoke.py` exists |
| L4 Tour and sim evaluation | `tour_lite` (seed 7, every PR), full `orchard_tour` (nightly, seeds 7 / 11 / 23), `readiness` (10 tours over 3 seeds); legacy seed × terrain matrix with `tools/headless_eval.py` + `tools/nandi_kpi.py` | capped runner in segments today; the CI runner for the full tour | [BUILT] tests; full tour only in segments (section 6) |
| L5 HIL / bench | One leg and the arm through the HAL ([hal-sim-to-real.md](hal-sim-to-real.md)); hardware e-stop bench test | bench | [PLANNED], blocked until bench hardware exists |

**Python version (correction).** The `atharv-nandi-sim` image runs **Python 3.10.12** (measured: `python3 --version` in the image, 2026-11-18). An earlier "ROS-free pytest, Python 3.11" rule and the README's "CI uses Python 3.11" do not describe the image; tests must stay 3.10-compatible (no `tomllib`, no `ExceptionGroup`).

**Tiers and markers** (`pytest.ini`, `--strict-markers`): unmarked = unit (headless, ROS-free, ≤ 10 s each); `gl` needs an offscreen GL backend and skips without one; `slow` ≤ 3 min each, nightly; `tour_lite` every PR; `tour` nightly; `readiness` before a demo. Per PR: `pytest -m "not slow and not tour and not readiness"`. Coverage gate ≥ 85 % line and ≥ 75 % branch on `atharv_nandi/core/`, new modules ≥ 90 % line (`.coveragerc`; the line and branch gates, and the new-module gate for the files given with `--new`, are read by `tools/coverage_gate.py` from a coverage JSON report, section 5.1; the sim image ships pytest-cov 6.3.0, measured 2026-11-18).

## 2 Proposed CI gates

All checks run on the CI runners, never heavy on a laptop (project rule), and only when a maintainer asks for a CI run. Names are proposals for the CI owner; nothing here changes CI.

| Gate | Command (inside the sim image) | When | Pass | Budget |
|---|---|---|---|---|
| G1 pytest unit | `python3 -m pytest -q -m "not slow and not tour and not readiness"` | every push | 0 failures | the unit tier took 666 s of runner time summed over the files (section 5); per file ≤ 180 s except `tests/test_tour_lite.py` (146 s) |
| G2 ruff | `ruff check sim tests tools rl_training` | every push | 0 findings | ruff is not in the sim image (run D-04, 2026-11-17); add it to the CI requirements first |
| G3 headless ROS smoke | start the stack headless, call `/nandi/reset_world`, then `/nandi/auto_harvest`, wait up to 60 s of sim time [TARGET], read `/nandi/game_status.in_basket_names` | every PR to `feat/nandi-pilot` | the target crop is in the basket (basket truth, not `harvest_status`) | under 180 s wall [ASSUMED] |
| G4 tour_lite | `python3 -m pytest -q -m tour_lite tests/test_tour_lite.py` | every PR | the WP9 tour pass definition (the 5 min time gate is a strict xfail, CR-WP9-02) | ≤ 3 min wall |
| G5 nightly | `python3 -m pytest -q -m "slow or tour"` (includes the interlock mutation check `tests/test_interlock_mutation.py`), then `tools/spot_tour_bench.py tour --program orchard_tour` and `--pace 0.9` | nightly | every `slow` test, tour pass over seeds 7 / 11 / 23, RTF ≥ 0.9 per 10 s window | ≤ 60 min [ASSUMED] |
| G6 docs | `tests/test_design_docs.py` and `tests/test_wp12_docs.py` (part of G1) | every push | sections 3 and 4 | < 5 s |
| G7 coverage | `python3 -m pytest -q -p pytest_cov -m "not slow and not tour and not readiness" --cov --cov-config=.coveragerc --cov-report=json:cov.json`, then `python3 tools/coverage_gate.py cov.json` | every PR | line ≥ 85 % and branch ≥ 75 % on `atharv_nandi/core/` (section 5.1) | one CI run; the capped runner needs 14 partial runs |

G3 is the gate that would have caught the live grasp bug: it reads the basket, so a false-positive grasp check fails it.

## 3 What `tests/test_design_docs.py` checks (WP0)

| Check | Rule |
|---|---|
| Files exist | the ten files of `docs/` listed in [README.md](README.md) |
| Relative links | every Markdown link target that is not a URL or an anchor resolves from the file's folder |
| No external plain HTTP | any `http://` is followed by `localhost` or `127.0.0.1` |
| No IP addresses | every dotted quad is `127.0.0.1` |
| Contract strings | every `/nandi/...` name in `docs/hld.md` appears as a string in `atharv_nandi/` or on a line marked `[PROPOSED]`; every name in `atharv_nandi/` is in `docs/hld.md`; rows marked `` `atharv_nandi/` [BUILT] `` are in code |
| Mermaid structure | each block starts with a known diagram type, balanced brackets and quotes per line, no `;`, `stateDiagram-v2` lines in `A --> B` or `A --> B: label` form, and every diagram state has a table row |
| Fidelity register | every row is rated `faithful`, `approximated` or `not simulated`, and the eight base items have a row |

It is a structural Mermaid check, not a render: `mmdc` is not in the sim image.

## 4 What `tests/test_wp12_docs.py` checks (WP12)

| Check | Rule |
|---|---|
| Contract fold-in | every fragment topic row is in [hld.md](hld.md) section 7 with the same WP and fragment link; no section 7 row without a fragment row; `` `atharv_nandi/` [BUILT] `` exactly when the name is in code, else `[PROPOSED]`; every `/nandi/*` name in code is in `hld.md` itself |
| OS 5.4 | every OS 5.4 contract addition has a status row in hld 7.1, and every `tele/alert` id is a string in `atharv_nandi/` |
| Assisted register | every assisted tag the code emits (AST scan of `assisted` lists, sets, keywords and functions under `atharv_nandi/`) has a row in [sim-fidelity-register.md](sim-fidelity-register.md) section 2; every OS 3.9 tag and every requested work-package row has one; a row marked `[BUILT]` names a tag that is in code |
| Safety case | at least 20 rated hazards, covering crown strike / snag, wind, the fall envelope, debris and bunches, blades and jaws, orbit near workers, laterals, animals, the local safety link and the beacon; the beacon table equals `hooks.beacon.PRIORITY` and `DEFAULT_PATTERNS`; section 8 cites 0.083 m, 0.114 m and the live `config/crown_gates.json` run id |
| State tables vs code | TREE 360 = `tree360.PHASES`; dock = `dock_logic.STATES` and `SESSION_STEPS` (order too); mission = `mission_logic.STATES`; WORK phases; recovery priority = `recovery_behaviours.PRIORITY`; engine `ACTIVE_STATES` |
| Test plan | every `tests/test_*.py` this plan names exists, or its row says `[PROPOSED]` |

## 5 Suite run on 1840118 (WP12, capped runner)

Command per file: `nandi-sim <worktree> python3 -m pytest -q -o addopts="" -p no:cacheprovider -m "not slow and not tour and not readiness" tests/<file>` (2 CPU / 2 GB, 180 s cap, no network, MuJoCo 3.3.7, Python 3.10.12; other agents' runs shared the host). 103 files (the 102 of 1840118 plus `tests/test_wp12_docs.py`): **2,757 passed (2,737 before WP12 + 20 new), 1 failed, 25 skipped (`gl` without a GL backend, optional `ompl`), 6 xfailed, 53 deselected** (`slow`, `tour`, `readiness`). The failure is the WP11 timing gate `tests/test_world_budget.py::test_lidar_sweep_mean[orchard_demo]` (safety sweep 4.69 ms, coverage sweep **5.38 ms** against the 5.0 ms gate on a contended host; WP11 measured 3.75 / 3.92 ms and called the gate "tight on demo", finding F-4). It is not a docs regression; WP12 changes no code. Slowest files: `tests/test_tour_lite.py` 146 s, `tests/test_physics_alignment.py` 43 s, `tests/test_world_budget.py` 17 s. The "~780 passed" figure in older notes is stale: it predates WP23-WP34.

### 5.1 Coverage and interlock mutation (WPCOV, capped runner, 2026-11-18)

Measured on `feat/nandi-wpcov` (base `feat/nandi-pilot` @ fb8c22c) in the capped runner (2 CPU / 2 GB, 180 s per run, Python 3.10.12, pytest-cov 6.3.0, coverage 7.16.2; no run id, so not a KPI under the run-identity rule; not the CI runner). Every test file of the per-PR selection except `tests/test_tour_lite.py` ran in 14 runs (`COVERAGE_FILE=.coverage.c<n>`, `--cov --cov-config=.coveragerc --cov-fail-under=0`); every `slow` test ran alone (41 runs); then `coverage combine`, `coverage json` and `python3 tools/coverage_gate.py`. `tests/test_tour_lite.py` exceeds the 180 s cap under tracing (it passes without coverage: 5 passed, 1 xfailed, 107 s), so `demo_tour.py` is only partly measured here.

| Selection | Tests | `atharv_nandi/core/` line | branch | coverage.py total (core + sim/world + sim/hooks) |
|---|---|---:|---:|---:|
| unit tier (per PR, minus tour_lite) | 2,838 passed, 25 skipped, 5 xfailed, 0 failed | **90.90 %** (27,764 / 30,542) | **84.79 %** (9,154 / 10,796) | 90.1 % |
| unit + `slow` (nightly, minus tour_lite) | + 39 passed, 2 skipped (`gl`) | **93.08 %** (28,428 / 30,542) | **87.18 %** (9,412 / 10,796) | 91.9 % |

Both gates hold (≥ 85 % line, ≥ 75 % branch). Before WPCOV a first unit-tier measurement on fb8c22c gave 89.99 % / 83.74 % (one of its 12 runs timed out on `tests/test_tour_lite.py` and kept partial data, so it is approximate). Lowest modules (unit + `slow`; the full per-module table is the output of `tools/coverage_gate.py`):

| module | stmts | line % | branches | branch % |
|---|---:|---:|---:|---:|
| `demo_tour.py` | 1505 | 42.6 | 460 | 37.8 |
| `tree360.py` | 1192 | 89.3 | 384 | 75.8 |
| `tasks/sim_executor.py` | 875 | 90.1 | 312 | 76.9 |
| `console_protocol.py` | 1298 | 90.3 | 600 | 81.5 |
| `presets.py` | 702 | 91.0 | 374 | 85.8 |
| `privacy_filter.py` | 387 | 91.7 | 100 | 79.0 |
| `tools/tool_changer.py` | 466 | 92.1 | 148 | 89.2 |
| `records.py` | 430 | 92.1 | 210 | 85.2 |
| `work_assist.py` | 863 | 92.5 | 344 | 84.0 |
| `camera_rig.py` | 228 | 93.4 | 36 | 80.6 |
| `geo_frame.py` | 187 | 93.6 | 68 | 91.2 |
| `ready_check.py` | 654 | 93.9 | 280 | 89.6 |

`demo_tour.py` is exercised by the `tour_lite` / `tour` tiers, `tasks/sim_executor.py` by the `slow` station tests (unit tier alone: 19.8 % / 4.8 %). WPCOV added unit tests for the adapters the unit tier never ran: `ros_client.py` 0 → 100 % (stand-in rclpy), `gui_widgets.py` 0 → 100 % (stand-in tkinter), `rl_balance_policy.py` 57 → 99 % (stand-in SB3), `scene_cull.py` 79 → 98 % (headless `mjv_updateScene`, no GL), `scope_kpi.py` 77 → 100 % (`tests/test_wpcov_adapters.py`).

**Interlock mutation check** (MVP-SPEC 6): `tests/interlock_mutants.py` lists 25 interlocks, each as the attribute(s) that implement it, a replacement that disables it and its guard test files. `tests/conftest.py` installs the mutant named in `NANDI_MUTANT` before collection; `tests/test_interlock_mutation.py` (`slow`) runs the guard files once per mutant in a subprocess and fails when they still pass. Covered: blade / jaw close, tool change, the live stability rule, crown raise, crown walk, static margin, protective field classification, field latch → e-stop latch, geofence, noVNC + actors (world build and Ready check), e-stop re-arm causes, fall detection, speed cap, orbit person pause, animal stop, hand in the WORK envelope, crown-scaled fields, bunch exclusion, follow-tag loss, both e-stop confirmations (the step machine and, separately, its `confirmed` backstop), self-right person gate, basket transfer, RTK start gate. "Share lower" belonged to the superseded WP3 vegetable harvester and has no code. Result: every mutant is caught (25 / 25; 28 tests incl. the 3 unit-tier registry checks, 14 s). Finding on the first run: disabling only the e-stop `confirmed` backstop was not caught by any test; `tests/test_wpcov_review.py` now guards it. **Fixer review (2026-11-18):** four mutants patched code the robot never ran. `animal_stop` renamed `CLS_ANIMAL`, which the classifier and the monitor both read, so nothing changed; it now empties `safety_fields.animal_hits`, the rule `FieldMonitor.update` latches on (caught by the MuJoCo dog / monkey / cattle actor tests too). `rtk_start_gate`, `self_right_person_gate` and `stability_static_margin` patched functions with no product caller while the live rules were copies: the Ready check now calls `gnss_model.rtk_start_gate` (with the preset's `float_ok`) and a named `_person_clear_confirm` check (patched together with the spike's `self_right_gate`), and `tool_changer.swap_gate`, `tool_changer.stability_ok` and the WP8 executor precheck call `stability.gate_static_margin`; their guards are now the live-path files (`tests/test_ready_check.py`, `tests/test_tool_changer.py`). `test_every_mutant_patches_something_the_product_uses` (unit tier, ast) fails when a mutant's targets are not loaded by any code in `atharv_nandi/`; `bunch_exclusion` is the one listed exception (`UNIT_ONLY`: the 3 m rule has no caller yet, open). The slow check also runs every guard set once without a mutant and requires it to pass, and requires a `FAILED` test under the mutant, so a flaky or broken guard file cannot count as a catch (25 / 25 caught, 16 baselines pass; 44 tests, 136 s in the capped runner). `stability_ok` now also refuses a `latched` or `degenerate` monitor and a `margin_m` published as None (the hook's encoding of NaN / -inf), and accepts numpy scalars.

**Review fixes** (each with a test in `tests/test_wpcov_review.py`): NaN / inf operator commands (a NaN yaw came out of `SpeedCap.clamp_cmd` as full yaw rate, an infinite vx as NaN), NaN tilt and a NaN geofence margin let an e-stop re-arm pass, `load_safety_config` returned the nested `DEFAULT_CONFIG` dicts (editing the loaded geofence edited the defaults for the process), a NaN speed dropped `d_stop` from the protective field, NaN margin / crown extension passed `swap_gate`, the blade-close gate moved from the MuJoCo hook to the pure `tool_changer.close_gate` with one `stability_ok` rule shared with the WP8 executor (a NaN or non-numeric margin passed one of the two copies), the `AuditLog` dedupe check-and-set raced across bridge threads, `PresetQueue` mission-id suffixes could collide, the control-panel client never dropped a timed-out service request, unclosed `json.load(open(...))` in `demo_tour.py` (now `run_log.read_json_dict`), dead code in `tool_changer.apply_ops` and `scope_kpi.tree360_kpis`.

## 6 Orchard mandatory tests (ORCHARD-MVP-SPEC 6)

| Test | Owner | File | Tier | Status on 1840118 (evidence in the fragment) |
|---|---|---|---|---|
| tour gates: every `orchard_tour.yaml` step and preset default against the 2.3 gate table, speed caps, slope rules | WP27 | `tests/test_tour_gates.py` | unit | pass (32 passed, 2 strict xfail: tour length CR-WP27-02) |
| crown gate matrix ext {0, 0.6, 1.6} × cross {0, 2, 3, 5}° × pitch {0, 5, 10}°, full layout, rack down / uphill, PARK and worst T4 | WP6 | `tests/test_crown_gates.py`, `tools/crown_gate_matrix.py` | unit (table check) + tool | pass: min 6-foot 0.132 m (> 0.13), run `wp6-matrix-20261117T203229Z-722981b` |
| e-stop sit-down at ext 1.6 on 5° | WP6 | `tests/test_estop.py`, `tools/crown_gate_matrix.py sitdown` | unit + tool | pass: no fall, CoM margin 0.162-0.238 m |
| crown walk 6 s at 0.15 m/s, ext 0.6 / 1.6, cross 0 / 2°, seeds 7 / 11 / 23: no auto-retract, tripod ≥ 0.03 m | WP6 | `tests/test_crown_gates.py` | `slow` | pass: 0 auto-retracts, min predicted tripod 0.052-0.058 m |
| gust 11 m/s mid-stride, ext 1.6, flat and 3°: stance ≤ 0.5 s, no fall | WP6 | `tests/test_wind_response.py` | `slow` | pass in the station test (stance 0.12-0.28 s); **the tour segment C2 measured 0.54 s from the truth onset** (O-18 fails there) |
| tripod-stance wind tipping speed, crown up | WP6 | `tests/test_wind_response.py`, `tools/crown_gate_matrix.py tipwind` | unit + measurement | 10.6 m/s step wind toward the rear (worst), 16.1-23.0 static |
| crown strike into a wood capsule and a foliage envelope: retract within 1 tick, tilt change ≤ 3° | WP24 | `tests/test_mast.py` | unit | pass: 0.012° / 0.011° |
| mast layout parity and dynamic parity at ext 0; ramp 0 → 1.6 → 0 per step | WP24 | `tests/test_mast.py` | unit | pass: qpos byte-identical over 2,000 steps; 0.400 mm max per-step delta |
| self-mask per extension | WP5 | `tests/test_pano.py`, `tests/test_privacy.py` | unit / `gl` | see the WP5 package notes |
| band coverage by stand-off d ∈ {0.9, 1.2, 1.5} m, 0-2.8 m with ≥ 0.3 m overlap | WP25 | `tests/test_coverage.py` | unit | see the WP25 package notes |
| coverage occlusion: a point behind foliage is not counted | WP25 | `tests/test_coverage.py` | unit | see the WP25 package notes |
| trunk fit with scaffolds from 0.6 m | WP5 | `tests/test_lidar.py` | unit | see the WP5 package notes |
| orbit radius RMS ≤ 0.10 m, rough + basins, seeds 7 / 11 / 23 | WP25 | `tests/test_tree360.py`, `tests/test_orbit.py` | `slow` | tour_lite: 0.029 m (truth); tour segment A: 2 re-entries at OT-1 (limit 1, open, WP25) |
| fruit pool: slot types, anchors, qpos0 fill, bumped mango settles ≤ 1 s, cut_k moves with the fruit, no fruit-wood collision, assign / recycle / reset | WP23 | `tests/test_fruit_pool.py`, `tests/test_trees.py` | unit | see the WP23 package notes |
| twist: a hanging papaya never releases in 10 s; closure without roll never releases; releases above τ with ≥ 30° roll, not at 0.8 τ | WP2 | `tests/test_twist_logic.py` | unit | see the WP2 package notes |
| tray: 0 fruit-on-fruit contacts; papaya placement ≤ 0.5 m/s | WP3 | `tests/test_transfer.py` | unit | see the WP3 package notes; tour_lite carry ≤ 0.42 m/s |
| cull: scene geom count after the post-filter; model unchanged | WP5, WP11 | `tests/test_world_budget.py`, `tests/test_detectors.py` | unit | pass: ≤ 700 scene geoms (stations max 287 on orchard_demo) |
| render geoms per tree ≤ 70 incl. leaf quads | WP23, WP11 | `tests/test_trees.py`, `tests/test_world_budget.py` | unit | pass: max 55 |
| sim_state mutation round trip over all providers | WP0 | `tests/test_sim_state.py` | unit | pass |
| records: TreeRecord ≤ 16 KiB, no image or personal keys; person events without coordinates; every MARK on the job sheet | WP9 | `tests/test_records.py`, `tests/test_uplink_contract.py`, `tests/test_job_sheet.py` | unit | pass; 0 contract violations over every tour run |
| lateral counter: 0 laterals stepped on during SCAN | WP4, WP9 | `tests/test_fixtures_orchard.py` (counter), `tests/test_demo_tour_unit.py` (KPI row `scan.laterals_stepped`) | unit | the counter is tested; the SCAN value is reported on the run card, not gated in a test |
| interlock mutation: orbit person-pause, animal stop, hand-in-envelope stop, crown-scaled fields, follow / lead tag loss, crown gate, self-right person gate, both e-stop confirmations | WP6, WP28, WP29, WPCOV | `tests/test_safety_fields.py`, `tests/test_recovery_behaviours.py`, `tests/test_work_assist.py`, `tests/test_task_interlocks.py`, `tests/test_interlock_mutation.py` (`tests/interlock_mutants.py`) | unit (+ `slow`) | pass: e-stop recovery over 172 event sequences, no motion before local → operator; WPCOV: 25 / 25 mutants caught (section 5.1) |
| RTH trigger tests (battery, wind, rain) with dock, charge, undock, resume ≤ 60 s | WP44 | `tests/test_rth_trigger_battery.py`, `tests/test_rth_trigger_wind.py`, `tests/test_rth_trigger_rain.py` [PROPOSED] | unit | phase C+, not built |
| run supervisor, RTH policy, hold points, teleop gate | WP38 | `tests/test_run_supervisor.py`, `tests/test_rth_policy.py` [PROPOSED] | unit | phase C+, not built |

### 6.1 Tour evidence (WP9, WP11; seed 7, capped runner, CI runner pending)

The full tour does not fit one 180 s run, so it runs as five fresh-world segments (A, B1, B2, C1, C2; `tests/test_tour_demo.py`, `tools/spot_tour_bench.py`).

| Gate | Limit | Measured | OK |
|---|---|---|---|
| RTF headroom, every 10 s sim window (198 windows) | ≥ 0.9 | min 2.93, median 5.18 | yes |
| RTF at the demo pace (segment B1 paced at 0.9) | every window ≥ 0.88 | 0.900 overall, window min 0.896 | yes (B1 only) |
| live `sim_runner` RTF while a tour runs beside it (C1) | ≥ 0.88 | window min 0.992, median 1.000 | yes |
| tour sim time | ≤ 18 min (1,080 s) | 1,986 s (segment sum, includes each segment's walk from home) | **no** (CR-WP27-02 / CR-WP9-02) |
| tour wall at the demo pace | ≤ 20 min | 2,207 s = 36.8 min | **no** |
| tour pass over the segments | all pass | B1, B2, C1 pass; **A fails** (TREE 360 OT-1 2 re-entries, limit 1); **C2 fails** O-18 (gust stance 0.54 s > 0.5 s) | **no** |
| culled capture p90, RGB + depth + seg at 320 × 240, crown aimed at the trunk | ≤ 300 ms | 403-427 ms | **no** (xfail F-1, WP5 / WP25) |
| `RenderBudget(0.35, burst 0.3 s)` per station | 0 skips | 11 of 48 skipped | **no** (xfail F-1) |
| physics on orchard_demo, robot at home | ≤ 0.50 ms/step | 0.102 | yes |
| SimDriver step with all 18 hooks | ≤ 1.0 ms | 0.349 | yes |

Reading: CPU is not what blocks the demo; the tour length and the render cost of real-world captures are. Seeds 11 and 23 and the CI-runner runs are pending (no CI unless a maintainer asks).

## 7 Pass/fail table per pilot criterion (legacy S1 criteria)

| Pilot criterion | Test level | Pass when | Fail when | Today |
|---|---|---|---|---|
| P-01 pick success pooled | L4 | >= 90 % [TARGET] | < 90 % | not measured |
| P-02 pick success worst config | L4 | >= 80 % [TARGET] | < 80 % | not measured |
| P-03 grasp false positives | L2 + L4 | 0 attempts with `grasp_reported_ok` true and the crop not in the basket [TARGET] | >= 1 such attempt (a drop after a passed check counts) | 1 observed live (not a recorded run), see [state-machines.md](state-machines.md) section 3.1; under this rule it fails P-03 |
| P-04 spurious reach aborts | L4 + L3 (G3) | 0 [TARGET] | >= 1 | 1 observed live (not a recorded run) |
| P-05 falls flat and ridges | L4 walking leg | 0 per 100 m over >= 100 m walked on each terrain [TARGET] | > 0 | not measured at the exposure floor |
| P-06 falls rough | L4 walking leg | <= 1 per 100 m over >= 300 m walked [TARGET] | > 1 per 100 m | not measured at the exposure floor |
| P-07 e-stop distance | L4 | max < 0.15 m [TARGET] | max >= 0.15 m | crown-up walking at 0.21 m/s: 3.0 cm (WP6, one run; the 10-trial matrix is not run) |
| P-08 e-stop time | L4 | max <= 500 ms [TARGET] | > 500 ms | stationary 0.28-0.42 s (WP6); crown-up walking 1.62 s to still with the driver's plain-gain hold (not the supervisor path) |
| P-09 throughput | L4 | >= 60 picks per sim hour [TARGET] | < 60 | not measured |
| P-10 energy per pick | L4 | a per-run value for every run with a pick; <= 1.2 x baseline median [TARGET rule] | missing, or over the ceiling | not measured |
| P-11 reset + harvest smoke | L3 (G3) | crop in basket within 60 s sim [TARGET] | not in basket | G3 not built |
| P-12 mission terminal states | L1 + L4 | 100 % with reason [TARGET] | any mission without a terminal state | L1 pass (`tests/test_mission_logic.py`) |
| P-13 provenance | L1 + L4 | 100 % simulated, `sim-` tenant [TARGET] | any payload without | L1 pass (`tests/test_kis_mapper.py`, `tests/test_uplink_contract.py`) |
| P-14 live RTF | L4 | mean >= 0.9 [TARGET] | < 0.9 | live `sim_runner` window min 0.992 during a tour segment (WP11, capped runner) |
| P-15 reproducibility | L4 | >= 8 of 9 configs agree [TARGET] | fewer | not measured |
| hardware e-stop requirement bench stop | L5 | 3 runs recorded in ms, power removed each time | any run where power stays | no hardware |

The orchard KPIs O-01 to O-23 (ORCHARD-MVP-SPEC 1.3) are scored per run by `atharv_nandi/core/scope_kpi.py` on the WP9 run card; their gates and the tour pass definition are in the WP9 package notes.
