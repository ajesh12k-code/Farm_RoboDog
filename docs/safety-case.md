# Safety case outline

Status: outline for review, not a safety approval. Gap covered: SAFE-01 (hazard analysis). WP12 (2026-11-18) adds the orchard hazards of ORCHARD-MVP-SPEC 5.2 and 5.3 (crown, wind, raised-mast fall envelope, debris and bunches, blades and jaws, orbit near workers, laterals, animals), the bystander beacon map (RTH-DOCKING E.11) and the tripod-margin correction, with the measured evidence of the work packages on `feat/nandi-pilot` @ 1840118. Nothing here allows outdoor motion: the hardware e-stop requirement (MUST, [DESIGNED]) calls for a hardware emergency stop, a speed cap and a geofence before any outdoor motion, and none of them exists in hardware.

All evidence quoted below is **simulated** (capped local runner, 2 CPU / 2 GB, MuJoCo 3.3.7, no CI run ids yet) unless it says otherwise. Simulation is never evidence of field safety (evidence rule); [sim-fidelity-register.md](sim-fidelity-register.md) says what each number cannot show.

## 1 Claim structure

| Level | Claim | Evidence that would support it | Status |
|---|---|---|---|
| C0 | Nandi is acceptably safe to run in a supervised orchard field trial | C1 to C5 | not claimed |
| C1 | Every hazard in section 3 is identified, rated and mitigated to an accepted level | this hazard log, reviewed by the hardware-engineer role | draft |
| C2 | A hardware stop removes motor power within the measured stopping time, whatever the software does | hardware e-stop bench test, 3 runs, ms recorded; no-software-override test with the runtime process killed | not started (no hardware) |
| C3 | The software never commands motion the operator did not allow | software latch, dead-man, speed cap, geofence, motion inhibit, local-link hold, e-stop recovery interlock tests | [BUILT] in sim (sections 3, 4, 6); the safety-relevant station tests pass on 1840118 (`-m slow`: `test_wind_response`, `test_task_interlocks`, `test_crown_gates`, `test_recovery_behaviours`, re-run for the WP12 review). WP12 did not re-run the full unit tier or the other station files (orbit, dock, TREE 360, tree stations, presets); their evidence is the owning WP's run |
| C4 | Simulation results are used only for what the simulator can show | [sim-fidelity-register.md](sim-fidelity-register.md) incl. the assisted register; risk: simulator treated as proof | draft |
| C5 | The raised crown never makes the robot less safe than the stowed robot: it is raised only when the gates allow, retracts on every trigger before the robot walks, and its fall envelope is covered by the fields | WP6 gate matrix, crown walk, gust and sit-down runs; WP24 strike and arbiter tests; crown-scaled fields | [BUILT] in sim; the field latency gap (HZ-15) and the C2 gate (sections 3 and 7) are open |

## 2 Rating scheme

Ratings are engineering judgements by the author, not measurements [ASSUMED all ratings]. A reviewer must re-rate them.

| Severity | Meaning | Likelihood | Meaning |
|---|---|---|---|
| S4 | death or permanent injury | L4 | expected in a season of use |
| S3 | injury needing treatment | L3 | plausible, has happened to similar machines |
| S2 | minor injury or robot damage | L2 | unlikely |
| S1 | nuisance, no injury | L1 | very unlikely |

Risk = severity x likelihood. 9 or more: not acceptable for a field trial. 4 to 8: acceptable only with the listed mitigations verified. Under 4: acceptable. Thresholds [ASSUMED].

## 3 Hazard log

Robot facts: 6 legs with 3 joints each, a dorsal gantry with 0.26 m travel each way, a 4-joint arm with two 0.34 m links and a 2-finger gripper (README, Robot Architecture); a joint-free 3-section crown mast with a 0.40 kg head (cap 0.6 kg), lens 1.09 m stowed to 2.69 m at full extension 1.60 m, rate 0.2 m/s; a tool rack with T2 clipper, T4 twist jaws and T5 pruner; an 8-cell desap tray; 38.5 kg at the full layout with a 3 kg basket (WP1, WP6 contracts). "SW" items are in code on `feat/nandi-pilot` [BUILT] unless marked; "HW" items are open.

### 3.1 Base hazards (vegetable-era log, re-checked)

| Id | Hazard | Cause | S | L (before) | Risk | Mitigations | L (after) | Residual |
|---|---|---|---|---|---|---|---|---|
| HZ-01 | Finger pinch at the gripper | a hand near a crop during `grasp`; gripper closes on a finger | S2 | L3 | 6 | SW: e-stop safe hold freezes the fingers; WORK warning field (hull + 1.0 m) stops the arm on any person return (WP6 `work_field`; hand at 0.6 m height: arm stop at 0.67 m, MEASURED); harvest starts only from a mission, preset or explicit operator command; HW (open): finger force limit by current cap, pads that cannot close below finger thickness [ASSUMED design] | L2 | 4 |
| HZ-02 | Pinch or crush at the gantry and arm joints | hand between gantry carriage and basket wall or between arm links during `carry` | S3 | L3 | 9 | SW: gantry slew limit 0.5 m/s (`harvest_logic.py`); e-stop safe hold; WORK warning-field arm stop (HZ-01); HW (open): guards over the rail, hardware e-stop removes arm power, torque limits | L2 | 6 |
| HZ-03 | Tip-over (stowed) | ridge or basin edge, soft soil, arm extended with load, push | S3 (onto a person) / S2 (robot) | L3 | 9 | SW: fall latch at 35° tilt sustained 0.3 s; walking limited to ≤ 5° cross and ≤ 5° tilt (WP6 `gate_walk_slope`); slopes > 5° refused by the Ready check (tour 10 b: refused `slope` at 10.0°, MEASURED); WP6 stability latch in WORK / STOPPED below 0.01 m for 0.3 s; tripod margin correction in section 8; HW (open): low centre of mass, stance width; sim cannot prove soil behaviour (cannot-prove list) | L2 | 6 |
| HZ-04 | Bystanders and children struck by a leg or the arm | people near the robot, children curious about a moving machine | S4 | L3 | 12 | SW: hull-relative protective and warning fields from the mast LiDAR (7 fans) and 2 low planes, stop and latch (WP6; adult, child, crouching, lying, ladder and pole pickers: protective latch with the person ≥ 0.19 m from the hull, MEASURED at 0.5 m/s); speed cap; geofence latch; handheld e-stop (WP1); beacon (section 5); procedure: exclusion zone 3 m [ASSUMED] with a spotter ([runbook-field.md](runbook-field.md)); HW (open): hardware e-stop on the robot and a wireless one with the spotter, bumper; person detection may only slow or stop (no-software-override rule) and never identifies anyone (privacy rule) | L1 | 4 |
| HZ-05 | Runaway: moves when not commanded or faster than commanded | stale command, software fault, RL policy output outside training range | S3 | L3 | 9 | SW: `cmd_vel` dead-man 0.5 s; gait params clamped; speed cap clamps commands (it does not latch); stale `sim_qpos` > 0.5 s stops the gait; `motion_inhibit` silent > 0.5 s = no inhibit (fails open: see residual); every hook drive relayed by the console stops on a stale drive (0.5 s) and on any STOP (WP10); RL mode only on explicit toggle; HW (open): speed cap in the motor controllers, hardware e-stop | L1 | 3 |
| HZ-06 | Comms loss while moving | Wi-Fi drop between operator surface and robot | S3 | L3 | 9 | SW: split into a **local safety link** and a **cloud console link** (WP7, hardware e-stop requirement, OQ-39 A). Local link silent > 0.5 s: stop, tools safe, crown retract, sit and hold (WP28 MEASURED: stopped 1.50 s, tools safe 0.46 s, crown stowed 8.36 s after the loss, drift 0.4 mm over 22 s). Cloud link silent > 5 s with the local link up: queue paused, 0 new targets, safe park ≤ 3 m to the alley edge, hold (moved 0.86 m, parked outside every canopy and basin footprint, MEASURED); resume only on an operator ack; sim_runner holds when `safety_status` is silent 1.0 s; HW (open): hardware e-stop that fails safe on radio loss, a real local link (dock Wi-Fi, LoRa or handheld) | L2 | 6 |
| HZ-07 | Unauthorised access to the controls | anyone on the network opens the web console or noVNC | S3 | L2 | 6 | [BUILT]: ports published on 127.0.0.1 only (`docker/run.sh`), host allow-list `NANDI_WEB_ALLOWED_HOSTS`, operator / viewer / dev roles, every console command operator-only and audit-logged (WP10); [DESIGNED]: field mode has no noVNC and an authenticated console; one device credential per robot for any broker link (device credential rule) | L1 | 3 |
| HZ-08 | Leg strike on a foot | person's foot inside the stance polygon | S2 | L3 | 6 | protective field on the low planes (0.18 m above the feet) and the bumper (50 Hz); exclusion zone; stopping distance target < 0.15 m (P-07) | L2 | 4 |
| HZ-09 | Battery fire or over-discharge | damaged or over-discharged pack | S3 | L2 | 6 | protected, fused pack, tray (battery safety rule); SW: reserve-based RETURN HOME (HZ-19); cell chemistry and thermal runaway are not simulated (fidelity register) | L1 | 3 |

### 3.2 Orchard hazards (ORCHARD-MVP-SPEC 5.2, 5.3; WP12)

| Id | Hazard | Cause | S | L (before) | Risk | Mitigations and evidence | L (after) | Residual |
|---|---|---|---|---|---|---|---|---|
| HZ-10 | **Crown strike or snag** | crown head driven into a scaffold, foliage, a prop rope or an overhead line while raising, orbiting or walking crown-up; a snag pulls the robot over | S3 | L3 | 9 | SW (WP24): sole mast writer with the arbiter (e-stop > safety > recovery > preset); raise refused unless the swept head sphere (r 0.120 m) clears registry wood, foliage envelopes and rope lines by ≥ 0.10 m and stays under `alley_clear_height_m`; strike check every physics step on every head geom (box, anemometer, GNSS puck); distance < 0.05 m stops upward motion, ≤ 0 submits a safety retract in the same substep. MEASURED: wood capsule strike at −0.9 mm, retract on the strike step, robot tilt change 0.012° (≤ 3°); foliage envelope 0.011°; rope at 1.6 m refuses the raise `crown_clearance`. HW (open): a mast that cannot be raised into a power line (height cap under the lowest line on site), a break-away or compliant head | L2 | 6 (the sim head is rigid and non-colliding, so a real snag force is not modelled: fidelity `geometric_crown_strike`) |
| HZ-11 | **Tip-over in wind with the crown up** | pre-monsoon gust hits a raised mast while walking or on a tripod stance | S3 | L3 | 9 | SW (WP6, WP7): raise refused at wind ≥ 10 m/s or a gust; crown-up walking only below 8 m/s; ordered response: 6-foot stance first (≤ 0.5 s), then retract; walking inhibited until stowed. MEASURED: 11 m/s gust at mid-stride, ext 1.6, flat and 3°: 6-foot stance 0.12-0.28 s after the stance request, retract 0.22-0.36 s after it, no fall, wind-loaded 6-foot margin ≥ 0.178 m through the retract. Tripod-stance tipping speed at ext 1.6: 16.1-23.0 m/s static, **10.6 m/s** under a step wind toward the rear (mast side), so the 8 m/s walking limit has 2.6 m/s of margin on the worst direction. **Open:** the tour segment C2 measured the stance **0.54 s** after the gust's truth onset (gate 0.5 s; WP9 / WP11, owner WP6 / WP7). No local wind data for the pilot region (OQ-13) | L2 | 6 (C2 open) |
| HZ-12 | **Raised-mast fall envelope** | the robot or the mast falls while the head is up; a person stands within the lens height | S4 | L2 | 8 | SW (WP6): fields scale with the crown while ext > 0: warning radius ≥ hull + lens height, protective ≥ hull + 0.5 × lens height (OS 5.2, ASSUMED sizes for this safety case); fall latch powers off; e-stop brakes the sections in place. MEASURED e-stop sit-down at ext 1.6 on 5° cross, legs de-powered 5 s: no fall, max tilt 7-9.5°, CoM margin to the ground-contact polygon 0.162-0.238 m; the belly rests do not reach the ground (≈ 5 cm short). Crown-up walking never crosses a basin rim (`gate_crown_path`; across a rim the predicted tripod went to −0.009 .. −0.022 m and the gates retracted). HW (open): mast lock for transport and after a fall (Fallen checklist, two-person lift) | L1 | 4 |
| HZ-13 | **Falling debris and bunches** | pruned shoots and cut fruit fall onto a worker; a human cutting a banana bunch next to the robot; a ladder picker dropping fruit onto the robot or a person beside it | S2 | L3 | 6 | SW: pruned wood drops as kinematic debris to the ground below the shoot (WP2 `kinematic_debris`); every contact task is in the pick band (z ≤ 0.8 m), nothing above 1.0 m is cut or picked: it becomes a MARK for a human (OS rule 11); banana bunches are MARK only (OQ-01 A), with a **3 m bunch exclusion** while a human cut is flagged (`safety_fields.bunch_exclusion`, `banana_tidy.bunch_exclusion_ok`) and the job sheet note "keep 3 m away while cutting"; ladder and pole pickers are field actors (protective latch, MEASURED). Procedure: no worker under the canopy section the robot is cutting | L2 | 4 (debris trajectories and impact are not simulated) |
| HZ-14 | **Blade and jaw pinch** | a hand inside the T2 clipper or T5 pruner blade, or the T4 jaws, at close or roll | S3 | L3 | 9 | SW (WP2, WP6, WP8): blade close and tool swaps blocked while the WORK warning field has a person (`work_field.person`, `blade_block`); e-stop actions open every blade and stop every tool on the latch edge; pruner refuses over capacity (> 12 mm, MARK `over_capacity`); T4 opens its jaws after ≥ 540° relative roll without release; tools open ≤ 3 s in SAFE PARK (else `tool_fault`). HW (open): blade guard, torque-limited jaws, mechanical lockout for tool changes by hand | L2 | 6 |
| HZ-15 | **Orbit near workers** | a TREE 360 orbit arc sweeps the hull toward a worker standing beside the tree or behind the canopy | S3 | L3 | 9 | SW (WP6, WP25): a person in the warning field **pauses** the orbit, the protective field stops and latches; the hull includes the planned arc ± 0.3 m; full orbits only where the registry and LiDAR show a clear annulus, else two-sided. MEASURED with the crown at 1.6 (lens 2.69 m): a person at 2.1 m pauses, at 1.1 m latches. **Open (spec owner):** the 10 Hz field tick lets a person one tick into the 0.30 m protective field (≥ 0.19 m left at 0.5 m/s); a latency term (closing speed × 0.1 s) or a faster tick is a change to L0 field sizes. The mast LiDAR sees through foliage, so a person hidden inside a neighbour canopy is not a cluster (fidelity `lidar_through_foliage`) | L2 | 6 |
| HZ-16 | **Drip laterals** | feet step on and damage drip laterals, or a foot snags in a lateral and the robot stumbles | S2 | L4 | 8 | SW (WP4, WP9): laterals are modelled as non-colliding query geoms; a foot within 2 cm of a lateral axis counts once per placement (`/nandi/world/state` `laterals`); SCAN reports `scan.laterals_stepped` with target 0 (run card); laterals are map-masked for the safety rays. Not avoided: there is no footstep planner (OS 7.1 risk 10, 7.2 out of scope), so the count is a KPI, not a mitigation. Procedure: laterals lifted or buried along the robot's alleys before a trial | L3 | 6 (foot snag forces not simulated: `noncolliding_laterals`) |
| HZ-17 | **Animals** | dog, monkey, cattle or nilgai approach or startle; the robot strikes or frightens an animal; an animal knocks the raised crown | S2 (animal or robot) / S3 (cattle against a person) | L3 | 9 | SW (WP6): animals are a separate class: a protective stop from the **warning** field (MEASURED: dog, monkey, cattle latch `animal` at > 0.30 m), never masked as persons and never handled by person logic; WP29 follow / lead modes never treat an animal as a tag holder. The class is truth-labelled in the sim (no animal detector exists). Procedure: no trial with loose cattle in the block | L2 | 6 |
| HZ-18 | **Wrong worker followed (WORK assist)** | a second person within 0.45 m of the tag is taken for the tag holder in follow / lead mode | S2 | L3 | 6 | SW (WP29): tags only for adult workers (`tag_not_adult` refused for a child actor), pseudonymous and rotated per shift; the WP6 protective latch still acts on everyone (WP29 never exempts a protective return); tag loss stops. Residual risk: association by distance only | L2 | 4 |
| HZ-19 | **Stranded low battery, two-person carry** | the pack runs out in the block; a 38.5 kg robot has to be carried out over basins | S3 | L2 | 6 | SW (WP7, WP28): reserve-based RETURN HOME (`energy_action: return_and_wait`), crown stowed first, row-following on odometry when RTK is lost. MEASURED: RTK FLOAT 0.165 m from home in 42.7 s; odometry only 0.054 m from home at 0.098 m/s; a stall on the way → assist. Battery CRITICAL tiers and RTH policy are phase C+ (WP38, [PROPOSED]) | L1 | 3 |
| HZ-20 | **Bystander misreads the robot's state** | workers in the row cannot see the console and walk up to a robot that is about to move | S3 | L3 | 9 | SW (WP1): the ring beacon on `mast_base` shows a state by priority, red only from the runner hold (section 5); a pre-motion flash before autonomous starts is a beacon state (`premotion`); audible tones are [PROPOSED] (WP43). HW (open): beacon brightness for daylight, sounder | L2 | 6 |

HZ-04 has the highest risk before mitigation (S4 x L3 = 12); HZ-12 is the only other hazard at the top severity S4 (risk 8, below the ten hazards at 9: HZ-02, HZ-03, HZ-05, HZ-06, HZ-10, HZ-11, HZ-14, HZ-15, HZ-17, HZ-20). Both S4 hazards are driven by hardware that does not exist (hardware e-stop, wireless e-stop, bumper, real LiDAR fields); together with C2 they keep C0 unclaimable until H1.

## 4 Stopping-time and stopping-distance method

### 4.1 In simulation (pilot criteria P-07, P-08)

1. Full stack started headless (no viewer) with `safety_supervisor` and `run_recorder`, on each terrain (`flat`, `ridges`, `rough`, and `orchard_demo` for the crown cases) and each seed (7, 11, 23 for the orchard tiers; 7, 8, 9 are the `headless_eval.py` defaults).
2. Walk forward with `cmd_vel` hold at `linear.x` 1.0 and `speed_scale` at the cap, 1.0 (`config/safety.yaml`; the gait itself allows 1.5) for 5 s [ASSUMED warm-up] so the gait is steady. Crown cases: ext 0, 0.6 and 1.6 at the crown-up walking speed (≤ 0.15 m/s).
3. At a random time inside the next gait cycle call `/nandi/estop`; record `t0` as the sim time of the latch.
4. Read the trunk position from `/nandi/sim_qpos` (every 0.02 s of sim time) until the trunk speed is below 0.02 m/s for 0.2 s (the WP6 `estop.stopped` metric; the base metric in `safety_status.last_stop` is the first sample below 0.02 m/s); `t1` is that time.
5. Stopping time = `t1 - t0`; stopping distance = horizontal trunk displacement between `t0` and `t1`.
6. 10 trials per configuration [ASSUMED]. Report the maximum, the median and the count over target, never only the mean.

Measured so far (WP6, capped runner): stationary P-08 0.28-0.42 s at ext 0 and 1.6 with the hold tuning kv × 10 and 4 N·m (`config/safety.yaml`); e-stop while walking crown-up at 0.21 m/s: sections hold at 1.6 m, tools safe, trunk still after **1.62 s and 3.0 cm** with the driver's plain-gain hold; tour segment C2: e-stop stationary 0.08 s and 0 motion during the hold. The 10-trial matrix of step 6 has not been run.

Resolution is one `sim_qpos` period, 20 ms of sim time. The sim measures how the gait controller stops, not how motors and gearboxes stop (servo torque and backlash are approximated, see the fidelity register).

### 4.2 On the bench (hardware e-stop requirement, no-software-override rule)

1. Robot on a stand, legs free, running at the speed cap; crown raised to 1.6 m for the crown-brake case.
2. Press the hardware e-stop. Record the time from the button contact opening to motor current reaching zero (a current probe or logic analyser on the motor supply) and to joint standstill (encoder, or video at 240 fps or more [ASSUMED]); for the crown, the head travel after the press.
3. Repeat 3 times (hardware e-stop requirement acceptance) with the runtime running, and 3 times with the runtime process killed first (no-software-override rule acceptance).
4. Record each run's stopping time in ms with device id, run id and date (run-identity rule).

## 5 Beacon map (bystander interface)

Workers in the row cannot see the console, so the beacon is the bystander interface (RTH-DOCKING E.11). The lamp is a ring on `mast_base` (world z 1.028-1.052 m), written by `atharv_nandi/sim/hooks/beacon.py`; patterns come from `config/beacon.yaml` or `$NANDI_BEACON_CONFIG` when present, else from the defaults below (`DEFAULT_PATTERNS`, colours ASSUMED). The state is chosen by priority, highest first, from the runner hold and the newest `/nandi/beacon/request`. `estop` and `fault` come only from the runner hold (`fault` for the hold reasons `hook_fault` and `supervisor_lost`) and can never be requested; the handheld e-stop does not colour the lamp until the supervisor has latched on it, so the lamp never claims a stop that has not happened. `tests/test_wp12_docs.py` checks this table against the code.

| Priority | Beacon state | Colour | Pattern | Hz | Operator state (E.11, Hindi) | Meaning for a worker |
|---|---|---|---|---|---|---|
| 1 | `estop` | red | steady | 0 | Stopped, रुका (Ruka) | latched e-stop, safe hold; do not approach until it is reset by the operator |
| 2 | `fault` | red | steady | 0 | Stopped, रुका (Ruka) | the runner holds because a hook failed or the supervisor went silent |
| 3 | `premotion` | amber | flash | 4 | pre-motion warning | the robot is about to start, restart or stand up: step away |
| 4 | `assist` | amber | flash | 1 | Stopped, रुका (Ruka) | the robot waits for a person (stuck, dock failed, tool fault) |
| 5 | `stopped` | amber | steady | 0 | Stopped, रुका (Ruka) | paused, holding or safe-parked; can move again after an operator ack |
| 6 | `charging` | cyan | breathe | 0.25 | Charging, चार्ज (Charge) | on the dock |
| 7 | `driving` | blue | steady | 0 | You are driving, आप चला रहे (Aap chala rahe) | a person drives it over the local link |
| 8 | `tool_change` | blue | steady | 0 | Working, काम (Kaam) | rack swap: the arm moves at the rack |
| 9 | `going_home` | green | pulse | 0.5 | Going home, घर जा रहा (Ghar ja raha) | returning home or docking |
| 10 | `working` | green | steady | 0 | Working, काम (Kaam) | running a preset |
| 11 | `idle` | amber | steady | 0 | Stopped, रुका (Ruka) | nothing requested |

Gaps: the E.11 audible cues (chirp, fault tone, rising tone) are not modelled ([PROPOSED], WP43); the beacon is render-only in the sim (fidelity register), so daylight visibility is unproven.

## 6 L0 safety layering

L0 hard safety (WORKFLOWS-SPEC A.0) is code, never configuration: the WP6 limits are constants in `atharv_nandi/core/stability.py` and `atharv_nandi/core/safety_fields.py` (`config/safety.yaml` only lists them), the WP24 arbiter cannot be bypassed by any source, and the WP34 engine checks `/safety/*` before every tick (e-stop: stop now, an open atomic block is logged `dropped`; protective field or local link lost after `start_ack`: freeze).

| Layer | What it enforces | Source |
|---|---|---|
| hardware (open) | e-stop removes motor power; speed cap in firmware; mast height cap | hardware e-stop requirement, section 7 |
| runner hold | every actuator held while the supervisor latches or is silent 1.0 s; crown sections braked | `sim_runner.py`, WP24 |
| supervisor latch | `operator`, `geofence`, `fall`, `handheld`, `protective`, `animal`, `bumper`, `stability`; reset refused while the cause is present | `safety_supervisor.py`, WP6 |
| motion inhibit | stance / walk / stop and `speed_scale` from stability and fields, plus recovery locks; honoured by `sim_driver` and `locomotion_service` | WP6, WP28 |
| crown gates and arbiter | raise gates (6-foot ≥ 0.13 m at the target, cross ≤ 3°, tilt ≤ 5°, wind < 10 m/s, no gust, clearance ≥ 0.10 m), crown-up walk gates (tripod ≥ 0.04 start / 0.03 continue, ≤ 0.15 m/s, wind < 8 m/s, slope ≤ 2°), retract triggers (wind, margin, tilt > 12° or filtered > 6°, link, battery) | WP6, WP24 |
| e-stop recovery | re-arm only after a local confirmation (local link only) and then an operator confirmation, checklist, crown retract; never resumes mid-cut (MEASURED over 172 event sequences: no motion before both) | WP28 |
| Ready check and tour gates | every preset and scripted tour step against the gate table, speed caps and slope rules | WP27 |

## 7 Open hardware items (hardware e-stop requirement and related)

| Item | Requirement | Status |
|---|---|---|
| Hardware e-stop that removes motor power | hardware e-stop requirement | [DESIGNED], no hardware |
| Speed cap in hardware or motor firmware | hardware e-stop requirement | [DESIGNED] |
| Geofence (needs a position source; RTK for rovers, position-source requirement COULD, RTK-fix rule) | hardware e-stop requirement | [DESIGNED]; RTK [PLANNED]. On orchard worlds the supervisor still uses the legacy geofence of `config/safety.yaml` (CR-WP10-01) |
| Bumper | no-software-override rule | [PLANNED]; modelled as a touch sensor (WP6) |
| Wireless (handheld) e-stop for the spotter | this document | modelled in the sim (WP1 `/nandi/handheld_estop`, WP6 latch); no hardware |
| Local safety link (dock Wi-Fi, LoRa or handheld) independent of the cloud console | OS 5.3, OQ-08, OQ-39 | modelled (WP7 heartbeats); hardware choice open |
| Crown brake and mast lock; mast height cap under overhead lines | this document (HZ-10, HZ-12) | proposed |
| Blade guard and tool lockout | this document (HZ-14) | proposed |
| Beacon brightness and a sounder | this document (HZ-20), RTH-DOCKING E.11 | proposed |
| Guarding of the gantry rail | this document (HZ-02) | proposed |

## 8 Tripod-margin correction (0.083 m)

The static tripod margin is the stability number every crown and walking gate rests on. Its history, so that nobody quotes the wrong one:

| When | Value | Model and basis | Source |
|---|---|---|---|
| MVP-SPEC 5.3 (vegetable era) | 0.11-0.13 m | computed, nominal | MVP design spec 5.3 |
| ORCHARD-MVP-SPEC 2.1 correction | **0.083 m** base robot | measured, probe, pre-alignment model | ORCHARD-MVP-SPEC 2.1, O-17, 7.1 risk 2 |
| WP0c, aligned model | 0.114 m base robot on flat ground | measured, `wp0c-matrix-…-96aadf8` | WP0c feasibility report |
| WP6 full layout (mast, 0.6 kg head cap, rack with T2 / T4 / T5, 3 kg basket) | flat crown-up **0.102 m at arm PARK, 0.088 m at the worst T4 pose** (1.8 kg fruit); 5° cross 0.060 / 0.045 m at ext 1.6; 10° rear-down with the rack uphill at the T4 pose: statically unstable (−0.004 .. −0.019 m, beyond the 5° walking limit) | measured, `wp6-matrix-20261117T203229Z-722981b` | the WP6 package notes |

Reading: 0.083 m was right for the probe model and is superseded by the aligned numbers, but the conclusion it drove still holds. The margin is thin. Crown-up walking is limited to ≤ 2° slope at ≤ 0.15 m/s with a 0.04 m start and 0.03 m continue gate, and the worst T4 pose binds the gate table. Any change to the T4 pose, the TCP or a mass needs a re-run of `tools/crown_gate_matrix.py matrix --write` (`test_matrix_rows_reproduce_live` fails when the table is stale).

## 9 What the simulator cannot show for this case

The crown is kinematic and rigid (no motor dynamics, sway or snag forces), the canopy is static (no wind or leaf motion), the LiDAR sees through foliage, actors are kinematic and truth-classified, laterals are non-colliding, soil is rigid with a soft-contact sink, and there is no battery chemistry. Each hazard above whose evidence depends on one of these says so in its residual column; the full list with consequences is the assisted register in [sim-fidelity-register.md](sim-fidelity-register.md) section 2.
