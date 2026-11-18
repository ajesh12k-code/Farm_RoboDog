"""Interlock mutation registry (WPCOV; MVP-SPEC 6 "Interlock mutation check", test-plan "interlock mutation").

Every safety interlock has one entry: the attribute(s) that implement it, a replacement that disables it (the
mutant) and the guard test files that must catch it. ``tests/conftest.py`` applies the mutant named in
``NANDI_MUTANT`` at ``pytest_configure`` (before any test module is imported, so ``from x import f`` in a test gets
the mutant too); ``tests/test_interlock_mutation.py`` runs the guard files once per mutant in a subprocess and fails
when they still pass (the interlock is unguarded).

A target is ``"module:attr"`` or ``"module:Class.attr"``. Patch the attribute where the code under test looks it up
at call time: a module global for module-level callers, or every importer that bound the name with ``from ... import``.
This file is ROS-free and imports nothing heavy at import time (targets are resolved only by ``apply``).
"""
import importlib
from typing import Any, Callable, Dict, List, NamedTuple, Tuple

ENV = "NANDI_MUTANT"


def _ok_decision(gate: str):
    def mutant(*_a, **_k):
        from atharv_nandi.core.interfaces import GateDecision
        return GateDecision.ok(gate)
    return mutant


def _const(value):
    def mutant(*_a, **_k):
        return value
    return mutant


def _zero_sectors():
    return {"front": 0.0, "side": 0.0, "rear": 0.0}


def _accept_any_confirm(self, t, kind, local_link_up=True, holding=True):
    """Mutant ``EstopRecovery.confirm``: any confirm, in any order, goes straight to the checklist."""
    self.confirms["local"] = self.confirms["operator"] = t
    self._to("checklist", t)
    return True, "mutant"


def _ready_ok(check_name: str):
    """Mutant Ready-check precondition: always passes."""
    def mutant(preset, ctx, it, *_a, **_k):
        it.ok(check_name)
    return mutant


class Mutant(NamedTuple):
    what: str                                     # the interlock, as named in the spec
    patches: Tuple[Tuple[str, Callable[[], Any]], ...]   # (target, factory of the replacement value)
    guards: Tuple[str, ...]                       # guard test files (unit tier: fast, ROS-free)
    spec: str = ""


MUTANTS: Dict[str, Mutant] = {
    # ---------------------------------------------------------------- MVP-SPEC 6 list
    "blade_close": Mutant(
        "blade / jaw close gate (person in WORK field, stability, no target)",
        (("atharv_nandi.core.tools.tool_changer:close_gate", lambda: _const(None)),),
        ("tests/test_wpcov_review.py", "tests/test_tool_changer.py"), "MVP-SPEC 2.3"),
    "tool_change": Mutant(
        "tool change swap gate (e-stop, holding, crown up, low margin, person)",
        (("atharv_nandi.core.tools.tool_changer:swap_gate", lambda: _ok_decision("tool_change")),),
        ("tests/test_tool_changer.py",), "MVP-SPEC 2.3, OS 2.3 rule 5"),
    "stability_live": Mutant(
        "live stability interlock (blade close and WP8 executor read the monitor margin)",
        (("atharv_nandi.core.tools.tool_changer:stability_ok", lambda: _const(True)),),
        ("tests/test_wpcov_review.py",), "MVP-SPEC 2.3"),
    "stability_gate_raise": Mutant(
        "crown raise stability gate",
        (("atharv_nandi.core.stability:gate_raise", lambda: _ok_decision("raise")),),
        ("tests/test_stability.py",), "OS 2.3"),
    "stability_gate_crown_walk": Mutant(
        "walk-with-crown-up stability gate",
        (("atharv_nandi.core.stability:gate_crown_walk", lambda: _ok_decision("crown_walk")),),
        ("tests/test_stability.py",), "OS 2.3"),
    "stability_static_margin": Mutant(
        "static CoM margin gate (live callers: tool_changer swap_gate + stability_ok, the WP8 executor precheck)",
        (("atharv_nandi.core.stability:gate_static_margin", lambda: _ok_decision("static_margin")),),
        ("tests/test_tool_changer.py",), "OS 2.3, MVP-SPEC 5.3"),
    "protective_field": Mutant(
        "protective / warning field classification (nothing is ever in a field)",
        (("atharv_nandi.core.safety_fields:FieldMonitor.classify", lambda: (lambda self, *a, **k: ([], []))),),
        ("tests/test_safety_fields.py",), "MVP-SPEC 5.3"),
    "field_latch_to_estop": Mutant(
        "a field latch engages the supervisor e-stop latch",
        (("atharv_nandi.core.safety_logic:SafetyCore.update_fields", lambda: (lambda self, now, fields: None)),),
        ("tests/test_safety_logic.py", "tests/test_estop.py", "tests/test_wpcov_review.py"), "MVP-SPEC 5.6"),
    "geofence": Mutant(
        "keep-in geofence (always inside)",
        (("atharv_nandi.core.safety_logic:geofence_margin", lambda: _const(1.0)),),
        ("tests/test_safety_logic.py",), "MVP-SPEC 5.6"),
    "novnc_actors_compose": Mutant(
        "noVNC + person actors lock-out at world build",
        (("atharv_nandi.sim.world.actors:novnc_up", lambda: _const(False)),),
        ("tests/test_actors.py",), "MVP-SPEC 5.4"),
    "novnc_actors_ready": Mutant(
        "noVNC + actors Ready-check no_go",
        (("atharv_nandi.core.ready_check:novnc_actor_conflict", lambda: _const(False)),),
        ("tests/test_wpcov_review.py", "tests/test_ready_check.py"), "MVP-SPEC 3.6, MP 0"),
    # ---------------------------------------------------------------- supervisor (WP6)
    "estop_reset_causes": Mutant(
        "e-stop re-arm refused while a cause is present",
        (("atharv_nandi.core.safety_logic:SafetyCore.causes_present", lambda: (lambda self, now=None: [])),),
        ("tests/test_safety_logic.py", "tests/test_estop.py"), "MVP-SPEC 5.6"),
    "fall_detect": Mutant(
        "tilt / fall detection",
        (("atharv_nandi.core.safety_logic:TiltFallDetector.update", lambda: (lambda self, tilt, fallen, now: False)),),
        ("tests/test_safety_logic.py",), "MVP-SPEC 5.6"),
    "speed_cap": Mutant(
        "operator command speed cap",
        (("atharv_nandi.core.safety_logic:SpeedCap.clamp_cmd", lambda: (lambda self, vx, vy, yaw: (vx, vy, yaw))),),
        ("tests/test_safety_logic.py", "tests/test_locomotion_logic.py"), "MVP-SPEC 5.6"),
    # ---------------------------------------------------------------- test-plan interlock row (WP6, WP28, WP29)
    "orbit_person_pause": Mutant(
        "orbit person pause (warning field)",
        (("atharv_nandi.core.safety_fields:WALK_WARNING", _zero_sectors),),
        ("tests/test_safety_fields.py",), "OS 5.2"),
    "animal_stop": Mutant(
        "animal stop (never person logic)",
        (("atharv_nandi.core.safety_fields:animal_hits", lambda: _const([])),),
        ("tests/test_safety_fields.py",), "OS 5.2"),
    "hand_in_work_envelope": Mutant(
        "hand inside the WORK envelope stop",
        (("atharv_nandi.core.safety_fields:WORK_PROTECTIVE", _zero_sectors),),
        ("tests/test_safety_fields.py",), "OS 5.2"),
    "crown_scaled_fields": Mutant(
        "crown-scaled protective field",
        (("atharv_nandi.core.safety_fields:CROWN_PROTECTIVE_K", lambda: 0.0),),
        ("tests/test_safety_fields.py",), "OS 5.2"),
    "bunch_exclusion": Mutant(
        "3 m bunch exclusion during a human cut",
        (("atharv_nandi.core.safety_fields:bunch_exclusion", lambda: _ok_decision("bunch_exclusion")),),
        ("tests/test_safety_fields.py",), "OS 5.2"),
    "follow_tag_loss": Mutant(
        "follow / lead tag loss stop",
        (("atharv_nandi.core.work_assist:WorkAssist._tag_gate", lambda: (lambda self, obs: None)),),
        ("tests/test_work_assist.py",), "MP 7"),
    "estop_both_confirmations": Mutant(
        "e-stop recovery needs a local then an operator confirm (both layers: the step machine and its backstop)",
        (("atharv_nandi.core.recovery_behaviours:EstopRecovery.confirm", lambda: _accept_any_confirm),
         ("atharv_nandi.core.recovery_behaviours:EstopRecovery.confirmed", lambda: property(lambda self: True))),
        ("tests/test_recovery_behaviours.py",), "MP 8"),
    "estop_confirm_backstop": Mutant(
        "e-stop recovery backstop: never past the checklist without both confirmations",
        (("atharv_nandi.core.recovery_behaviours:EstopRecovery.confirmed", lambda: property(lambda self: True)),),
        ("tests/test_wpcov_review.py",), "MP 8"),
    "self_right_person_gate": Mutant(
        "self-right person gate (live: the SELF-RIGHT Ready check; the spike tool's self_right_gate)",
        (("atharv_nandi.core.ready_check:_person_clear_confirm", lambda: _ready_ok("person_clear_confirm")),
         ("atharv_nandi.core.recovery_behaviours:self_right_gate", lambda: _const((True, "")))),
        ("tests/test_ready_check.py",), "MP 8"),
    "transfer_interlock": Mutant(
        "basket transfer / unload interlock (e-stop, arm PARK, gait stopped)",
        (("atharv_nandi.core.tools.kinematic_transfer:interlock", lambda: _ok_decision("transfer")),),
        ("tests/test_transfer.py",), "MP 9"),
    "rtk_start_gate": Mutant(
        "RTK start gate (live: ready_check._common)",
        (("atharv_nandi.core.gnss_model:rtk_start_gate", lambda: _const((True, ""))),),
        ("tests/test_ready_check.py",), "MP 0"),
}

# MVP-SPEC 6 names every interlock the nightly check must cover; "share lower" belonged to the superseded WP3
# vegetable harvester (superseded) and has no code in the orchard MVP.
SPEC_REQUIRED = {
    "blade close": ("blade_close",),
    "tool change": ("tool_change",),
    "stability gate": ("stability_live", "stability_gate_raise", "stability_gate_crown_walk", "stability_static_margin"),
    "protective field": ("protective_field", "field_latch_to_estop"),
    "geofence": ("geofence",),
    "noVNC + actor": ("novnc_actors_compose", "novnc_actors_ready"),
}
# Interlocks whose rule exists only as a pure function with unit tests: no product code calls it yet, so the mutant
# proves the unit test, not the robot. test_interlock_mutation checks this list both ways (a listed one that gains a
# live caller must leave it; any other mutant must patch code the product uses).
UNIT_ONLY = {
    "bunch_exclusion": "OS 5.2 3 m bunch exclusion: safety_fields.bunch_exclusion has no caller (no human-cut flag "
                       "in the WORK-assist path yet); the job sheet only prints the 3 m note",
}

TEST_PLAN_REQUIRED = ("orbit_person_pause", "animal_stop", "hand_in_work_envelope", "crown_scaled_fields",
                      "follow_tag_loss", "self_right_person_gate", "estop_both_confirmations")


def resolve(target: str) -> Tuple[Any, str]:
    """``module:attr`` / ``module:Class.attr`` -> (owner object, attribute name). Raises if missing."""
    mod_name, _, path = target.partition(":")
    owner: Any = importlib.import_module(mod_name)
    parts = path.split(".")
    for p in parts[:-1]:
        owner = getattr(owner, p)
    if not hasattr(owner, parts[-1]):
        raise AttributeError(f"{target}: no attribute {parts[-1]!r}")
    return owner, parts[-1]


def apply(name: str) -> List[str]:
    """Install mutant ``name`` (process-wide, never undone). Returns the patched targets."""
    if name not in MUTANTS:
        raise KeyError(f"unknown mutant {name!r}; known: {sorted(MUTANTS)}")
    done = []
    for target, factory in MUTANTS[name].patches:
        owner, attr = resolve(target)
        setattr(owner, attr, factory())
        done.append(target)
    return done
