"""Reason codes for workflows: ONE enum file (WP34; WORKFLOWS-SPEC A.3, MISSION-PRESETS 0.5).

The MP 0.5 base enum, the CHORES D.0 and RTH-DOCKING E.8 additions, and the codes the WP34 engine itself
raises. Other packages extend this file by change request, collected at each phase boundary. The validator
refuses any ``skip(reason)``, ``halt(reason)``, ``reason_on_fail`` or ``then: skip(reason)`` whose reason is
not in ``REASONS``.
"""
from typing import FrozenSet

# MISSION-PRESETS 0.5
MP_05 = (
    "out_of_reach_marked", "orbit_blocked", "range_lost", "low_light", "disk_low", "payload_full",
    "tool_unavailable", "rtk", "energy_reserve", "stability", "crown_gate", "crown_clearance", "wind", "slope",
    "geofence", "person_hold", "animal_hold", "local_link_lost", "cloud_link_lost", "stuck", "sink",
    "over_capacity", "paste_required", "twist_failed", "no_clearance", "bagged", "leaf_minimum", "season", "rain",
    "wet_foliage", "heat", "self_right_failed", "registry_mismatch", "not_in_mvp",
)

# CHORES D.0
CHORES_D0 = (
    "band_missing", "trap_missing", "lure_due", "storm_hold", "basin_inaccessible", "glean_out_of_reach",
    "tote_full", "coverage_partial", "electrical_hazard", "low_visibility", "not_trafficable",
    "animal_hold_timeout", "payload_disturbed",
)

# RTH-DOCKING E.8
RTH_E8 = (
    "dock_failed", "dock_occupied", "dock_tag_lost", "fiducial_mismatch", "charger_fault", "charge_thermal_hold",
    "dock_service_due", "dock_wet", "dock_dirty", "under_belly_blocked", "grid_outage", "workflow_changed",
    "rth_battery", "rth_wind", "rth_rain", "rth_heat", "rth_cell_temp", "rth_storm_alert", "rth_lightning",
    "rth_visibility", "rth_done", "rth_chore_done", "rth_tote_full", "rth_skip_ratio", "rth_cloud_loss",
    "rth_operator", "teleop_link_required",
)

# WP34 engine and WORKFLOWS-SPEC A.2-A.4 (the spec example uses ``skip(arc_blocked)``)
ENGINE = (
    "arc_blocked",           # WORKFLOWS-SPEC A.2 example (tree360 station re-entry failed)
    "timeout",               # an action ran out of ACTIVE time (clock.py)
    "primitive_failed",      # an action failed with no reason_on_fail
    "retry_exhausted",       # retry {n} used up
    "check_failed",          # check / wait_for else: fail
    "operator_abort",        # abort command (L1)
    "operator_pause",        # pause command (L1)
    "operator_skip",         # skip_target command or assist reply Skip
    "operator_halt",         # assist reply Halt
    "operator_takeover",     # assist reply Take over
    "assist_timeout",        # no assist reply within the timeout
    "start_ack_missing",     # start_ack timed out: no supervisor acknowledgement / local link down
    "estop",                 # L0 e-stop
    "protective_stop",       # L0 protective field
    "dropped",               # L0 stop inside an atomic block released the item (A.2)
    "held_at_estop",         # L0 stop inside an atomic block, item still held (A.2)
    "holding_item",          # release_safe could not place: keep holding, assist (A.3)
    "guard_violation",       # a YAML guard tripped with no more specific reason
    "drift_gate",            # resume: pose not within 0.10 m / 5 deg of last_safe after return_to_checkpoint
    "not_resumable",         # resume refused (no checkpoint, ...)
    "ready_check",           # resume: Ready check re-run failed
    "safe_park",             # request(safe_park)
)

REASONS: FrozenSet[str] = frozenset(MP_05 + CHORES_D0 + RTH_E8 + ENGINE)

# guard ``on_violation`` (WORKFLOWS-SPEC A.2): every one is at least as strict as the L0/L1 pause
ON_VIOLATION: FrozenSet[str] = frozenset({"pause", "stow_continue", "skip_target", "rth", "safe_park", "halt_latch"})

# run terminal states (engine.py)
TERMINAL: FrozenSet[str] = frozenset({"succeeded", "failed", "aborted", "halted", "estopped", "rth"})


def is_reason(code: str) -> bool:
    return isinstance(code, str) and code in REASONS


def check(code: str) -> str:
    """``code`` if it is in the enum, else ``ValueError``."""
    if not is_reason(code):
        raise ValueError(f"unknown reason code {code!r} (atharv_nandi/core/workflow/reasons.py)")
    return code
