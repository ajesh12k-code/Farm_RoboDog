"""Atharv Nandi workflow layer (WP34; WORKFLOWS-SPEC, the WP34 contract notes).

Predefined actions, presets, chores, recovery and missions as versioned ``nandi.workflow/1`` YAML behaviour
trees, compiled and ticked by a py_trees 2.2.3 runtime (OQ-33).

    schema.py        structural validator (hand-rolled; jsonschema is not in the image)
    compiler.py      YAML -> IR; library of areas (INDEX.yaml), subtree / template expansion, id@version#sha8
    validate.py      semantics + static check + versioning; CLI ``python3 -m atharv_nandi.core.workflow.validate``
    static_check.py  abstract interpretation (start_ack dominance, commit points / atomic, holding, caps, ...)
    engine.py        py_trees runtime, L0/L1 mirror, pause / resume / abort, checkpoints, fault matrix (needs py_trees)
    blackboard.py    namespaces and py_trees key access control
    clock.py         active-time clock (timeouts never count pauses)
    checkpoint.py    checkpoints + the ``workflow_checkpoint`` sim-state provider
    reasons.py       the one reason-code enum
    primitives/      catalogue v1 (registry.py), stubs for every primitive (stubs.py); WP35 / WP38 adapters

Only ``engine.py`` (and the run-time half of ``blackboard.py``) imports py_trees, so validation runs without it.
"""
SCHEMA = "nandi.workflow/1"
PY_TREES_PIN = "2.2.3"
