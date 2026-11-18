# Field runbook

Status: draft. Part 1 is for the simulator today [BUILT commands]. Part 2 is the field procedure for a future robot [DESIGNED]; it cannot be used until hardware gates HG1 and HG2 have passed and the hardware e-stop requirement (hardware e-stop, speed cap, geofence) is met. No outdoor motion before then.

## Part 1: simulator

### 1.1 Bring-up

| Step | Command or action | Expect |
|---|---|---|
| 1 | `docker/run.sh` from the repository root (README, Quick Start) | image `atharv-nandi-sim` builds once, container `atharv-nandi` starts with the repo at `/ws` |
| 2 | open <http://localhost:8080> | the 3D view, status panel and score |
| 3 | optional: <http://localhost:6080/vnc.html?autoconnect=1&resize=scale> | the MuJoCo viewer and Tk console |
| 4 | world knobs: `NANDI_CROPS`, `NANDI_TERRAIN`, `NANDI_SEED` before `docker/run.sh` | the world described in the README table |
| 5 | RL mode: `docker/run_rl.sh`, then Switch Gait (`G`) | `/nandi/locomotion_status.mode` = `rl` only when a policy is loaded |

Note: `docker/run.sh` removes any existing container named `atharv-nandi` first (code: `docker rm -f atharv-nandi`). Do not run it while someone else is using a live `atharv-nandi` session.

### 1.2 Pre-flight (sim)

- Status panel shows `rtf` close to 1 and `falls` 0.
- `/nandi/harvest_status.mode` is `rl_balance`, phase `idle`.
- Press `R` (Reset World) once so the timer, falls and score start from zero.

### 1.3 Stopping

| Need | Action today | After the safety package |
|---|---|---|
| Stop walking | `Space`, or release the movement keys; the gait also stops itself 0.5 s after the last held command (code: `locomotion_logic.py`) | `/nandi/estop` [PROPOSED] (the console STOP) latches with reason `operator`: every actuator holds, locomotion refuses `cmd_vel`, an active mission ends `failed` / `estopped`; `/nandi/estop_reset` re-arms, and refuses while a `fall` or `geofence` cause is present ([state-machines.md](state-machines.md) section 5) |
| Stop the arm | `Esc` (Release Arm) aborts a running harvest and sends the arm to rest | the same latch freezes the arm where it is (it does not send it to rest). **Do not reset the e-stop while the arm command and the measured arm pose differ.** Until the snap-to-measured-pose fix proposed in [state-machines.md](state-machines.md) section 4 lands, the harvest node keeps moving its command during the hold: a running harvest keeps stepping, and `Esc` (or a mission abort, which calls `release_arm`) slews the command to rest. `/nandi/estop_reset` then drives the arm in one jump from the held pose to that command (read from the code, not tested [UNVERIFIED]; pinch hazard HZ-02, [safety-case.md](safety-case.md)). `Esc` before or after the reset does not prevent the jump. If the arm was moving when the latch engaged, keep everyone clear of the arm and gantry before any reset, and treat the reset as an arm motion |
| Stop everything | `Ctrl+C` in the terminal running `docker/run.sh`, or close the MuJoCo viewer | unchanged |

### 1.4 Recovery

| Symptom | Cause (known) | Recovery |
|---|---|---|
| `falls` went up, robot on its side | tilt above 50 deg or trunk below 0.12 m (code: `game_status.py`) | `R` (Reset World). There is no self-righting. Note seed, terrain and what you were doing for the run log |
| `Drop missed: crop_item_k not in basket` | the crop was not in the gripper during carry, although the grasp check passed (the false positive in [state-machines.md](state-machines.md) section 3.1) or it bounced out | the arm is already at rest; walk so the crop is within reach and press `H` again; if it repeats on the same crop, `R` and report the seed and terrain |
| `Arm path out of reach (N cm off)` | the per-waypoint IK tolerance (3 mm, code) is stricter than the reach check | walk 10 to 20 cm closer [ASSUMED] and press `H` again; the harvest package widens the tolerance [PENDING-MERGE] |
| `crop_item_k out of arm reach (X.XX m), move closer` | crop inside 1.08 m but beyond the arm (about 0.83 m ahead at ground level) | walk closer |
| `... under the body - back up` | the nearest crop is under the trunk | back up (`S`) |
| `Previous harvest still stopping - try again` | an aborted harvest thread is finishing | wait 1 to 2 s [ASSUMED] and press `H` again |
| `No RL policy loaded - using Tripod gait` | default image or no archive in `rl_training/models/` | use `docker/run_rl.sh` with a trained archive (README) |

### 1.5 Collecting runs and logs

- Node output goes to the terminal running `docker/run.sh` (the launcher starts every node in the foreground process group, code: `launch/launch_nandi.sh`); copy it into the run report.
- noVNC log: `/tmp/novnc.log` inside the container (code: `docker/start.sh`).
- Recorded runs: `runs/<run_id>/` in the repository (bind-mounted, so visible on the host) once `run_recorder` and `tools/headless_eval.py` exist [PROPOSED]; summarise with `tools/nandi_kpi.py` [PROPOSED]. `runs/` is not committed.
- Every number taken from a run carries the run id, the date and the machine (run-identity rule).

### 1.6 Update and rollback

| Action | How |
|---|---|
| Update code | `git pull` on the integration branch; restart the container (code is bind-mounted, README) |
| Update the image | only when `docker/Dockerfile` changed: rerun `docker/run.sh` (it rebuilds) |
| Roll back | `git checkout <previous sha>` and restart; for an image change, rebuild at that sha |
| Check after any change | the unit tests (README, Tests) and one reset + harvest in the browser |

## Part 2: field procedure [DESIGNED]

Applies only after HG2 and once the hardware e-stop requirement is met. Kept short on purpose; it is reviewed again before first use.

### 2.1 Bring-up

1. Battery in its tray, fuse checked (battery safety rule).
2. Hardware e-stop pressed (latched) before power-on.
3. Robot computer boots the field-mode stack ([hld.md](hld.md) section 5): no noVNC, console on localhost or authenticated, `safety_supervisor` running.
4. Confirm `safety_status` (fields as in [hld.md](hld.md) section 3.1): `estopped` true with a `latched_reason`, `geofence_ok` true, `fallen` false, `stale` empty, and every entry of `nodes` ok. There is no comms field: the status does not report operator-surface comms loss (gap, [safety-case.md](safety-case.md) HZ-06), so check the operator link by hand with a stop test.

### 2.2 Pre-flight checklist

| Check | Pass |
|---|---|
| Exclusion zone marked, 3 m [ASSUMED] around the work area | marked, no people inside except the operator |
| Spotter with the wireless e-stop | present, tested once by pressing and releasing |
| Children and animals | none in or near the area; stop immediately if any arrive (HZ-04) |
| Speed cap set | value recorded in the run log |
| Geofence loaded | matches the marked area |
| Position fix | RTK fixed before a mission (RTK-fix rule) |
| Stop test | one e-stop press while standing removes motor power |

### 2.3 E-stop use

- Anyone may press an e-stop at any time, for any reason. No one is blamed for a stop.
- After a stop: keep it latched, check people first, then the robot; release only when the area is clear; reset with `/nandi/estop_reset` from a local surface.

### 2.4 Recovery from a fall

1. E-stop latched, motor power off.
2. Two people lift the robot by the frame, not by the arm or gantry (pinch, HZ-02).
3. Inspect legs and arm for damage; log the fall with time, terrain and run id.
4. Power on latched, check `joint_states` against the rest pose, then clear the latch.

### 2.5 Logs and update

- Copy `runs/<run_id>/` and the system logs from the robot computer to the evaluation machine after each session; nothing with people in frame leaves the robot unblurred (privacy rule).
- Updates never install during a mission or below half battery (the same rule the edge-unit platform sets for its units); keep the last good image for rollback.
