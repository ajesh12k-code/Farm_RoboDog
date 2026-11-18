# Atharv Nandi design set (pilot)

This folder is the design set for taking Atharv Nandi from a playable simulator to a pilot decision: architecture, state machines, safety, testing, simulator fidelity, the hardware abstraction and the field runbook.

None of these documents is approved. They are drafts for review.

## Reading order

| # | Document | What it answers |
|---|---|---|
| 1 | [hld.md](hld.md) | Node graph, full `/nandi/*` topic and service table, data flow to `kis/v1`, deployment |
| 2 | [state-machines.md](state-machines.md) | Mission, harvest phases, arm arbitration, e-stop latch (Mermaid + transition tables) |
| 3 | [safety-case.md](safety-case.md) | Hazard log, mitigations, open hardware e-stop items, stopping-time method |
| 4 | [test-plan.md](test-plan.md) | Unit / sim / HIL levels, proposed CI gates, pass/fail per pilot criterion |
| 5 | [sim-fidelity-register.md](sim-fidelity-register.md) | What the simulator models faithfully, approximately, or not at all, and what it cannot prove |
| 6 | [hal-sim-to-real.md](hal-sim-to-real.md) | Hardware abstraction target, migration from `/nandi/sim_qpos`, noise/latency layer, Mac GPU finding |
| 7 | [runbook-field.md](runbook-field.md) | Bring-up, pre-flight, e-stop use, recovery, collecting runs and logs, update and rollback |
| 8 | [cap-libraries.md](cap-libraries.md) | OMPL arm planning with MuJoCo collision checks, ros2_control-style HAL, contact/grasp pairs |

## Labels used in every document

Status of a thing:

| Label | Meaning |
|---|---|
| `[BUILT]` | In code on `feat/nandi-pilot`, with a path |
| `[PENDING-MERGE]` | In a work-package branch that has not merged into `feat/nandi-pilot`, with the branch |
| `[PROPOSED]` | A contract name (topic, service, tool, file) that a parallel work package is building; it is not in code on `feat/nandi-pilot` yet and may be renamed |
| `[DESIGNED]` / `[PLANNED]` | Written down in a design document, no code |
| `[UNVERIFIED]` | A claim nobody checked in code or in a cited document |

Status of a number (every measured number carries its run id, device id and date):

| Tag | Meaning |
|---|---|
| `[TARGET]` | A pilot target. Not a measurement |
| `[ASSUMED]` | An assumption made to plan with. Replace it with a measurement |
| `[MEASURED <run>]` | Measured in a named run; the run is described where it is first used |
| `(code: <path>)` | Read from a constant in this repository |
| `(doc: <id or path>)` | Taken from a cited document |

There are no invented measurements in this folder. Every pilot KPI is `[TARGET]` until `tools/nandi_kpi.py` [PROPOSED] prints it from a recorded run.

## Sources

The repository contract itself (topics, services, nodes) is in the top-level [README](../README.md); code wins over these documents when they disagree.
