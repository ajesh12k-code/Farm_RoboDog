"""Interlock mutation check (WPCOV; MVP-SPEC 6, docs/test-plan.md "interlock mutation").

For every interlock in ``tests/interlock_mutants.py`` a subprocess runs its guard test files with the interlock
disabled (``NANDI_MUTANT=<name>``, applied by ``tests/conftest.py``). At least one guard test must fail; if they all
pass the interlock is unguarded and this check fails. The per-mutant runs are ``slow`` (nightly, MVP-SPEC 6); the
registry check is unit tier (every PR) so a renamed interlock cannot silently drop out of the nightly check.
"""
import ast
import functools
import glob
import os
import subprocess
import sys

import pytest

import interlock_mutants as im

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UNIT_TIER = "not slow and not tour and not readiness"


def test_registry_targets_resolve_and_guards_exist():
    for name, mu in im.MUTANTS.items():
        assert mu.patches and mu.guards, name
        for target, factory in mu.patches:
            owner, attr = im.resolve(target)                      # raises when the interlock was renamed / moved
            assert getattr(owner, attr) is not None, target
            factory()                                             # the replacement builds
        for g in mu.guards:
            assert os.path.isfile(os.path.join(REPO, g)), (name, g)
            assert g != "tests/test_interlock_mutation.py", name


def test_registry_covers_the_spec_lists():
    for what, names in im.SPEC_REQUIRED.items():
        assert names, what
        for n in names:
            assert n in im.MUTANTS, (what, n)
    for n in im.TEST_PLAN_REQUIRED:
        assert n in im.MUTANTS, n


def _loads_in_package():
    """Every name / attribute *loaded* by code in atharv_nandi/ (ast: comments, docstrings, ``def`` names and plain
    assignments do not count)."""
    names = set()
    for p in glob.glob(os.path.join(REPO, "atharv_nandi", "**", "*.py"), recursive=True):
        with open(p, encoding="utf-8") as f:
            tree = ast.parse(f.read(), p)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                names.add(node.attr)
    return names


def test_every_mutant_patches_something_the_product_uses():
    """A mutant must disable code the product runs, not a function only its unit tests call (WPCOV review: the RTK
    start gate, self-right gate and static-margin gate were dead copies of the live rules). At least one patched
    attribute of each mutant is loaded by code in atharv_nandi/ (a call or reference, not its definition)."""
    loads = _loads_in_package()
    assert set(im.UNIT_ONLY) <= set(im.MUTANTS)
    for name, mu in im.MUTANTS.items():
        attrs = [t.partition(":")[2].rpartition(".")[2] for t, _f in mu.patches]
        live = any(a in loads for a in attrs)
        if name in im.UNIT_ONLY:
            assert not live, f"mutant {name!r} now has a live caller: drop it from UNIT_ONLY"
        else:
            assert live, f"mutant {name!r}: no product code uses {attrs} (only its unit tests would catch it)"


def test_apply_rejects_unknown_mutants():
    with pytest.raises(KeyError):
        im.apply("no_such_interlock")
    with pytest.raises((AttributeError, ImportError)):
        im.resolve("atharv_nandi.core.safety_logic:NoSuchThing.attr")
    with pytest.raises(AttributeError):
        im.resolve("atharv_nandi.core.safety_logic:no_such_function")


def run_guards(name, timeout: float = 150.0, guards=None) -> subprocess.CompletedProcess:
    """Run the guard files of mutant ``name`` (None = no mutant; ``guards`` overrides the file list)."""
    env = dict(os.environ)
    env.pop(im.ENV, None)
    if name is not None:
        env[im.ENV] = name
    env.pop("COVERAGE_PROCESS_START", None)
    files = guards if guards is not None else im.MUTANTS[name].guards
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-rf", "-p", "no:cacheprovider", "-o", "addopts=", "-m", UNIT_TIER,
           *files]
    return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)


@functools.lru_cache(maxsize=None)
def baseline(guards: tuple) -> subprocess.CompletedProcess:
    """The guard files with no mutant (cached per guard set): they must pass, or a mutant 'caught' by them proves
    nothing (a flaky or broken guard file would report every mutant naming it as caught)."""
    return run_guards(None, guards=guards)


@pytest.mark.slow
@pytest.mark.parametrize("guards", sorted({mu.guards for mu in im.MUTANTS.values()}))
def test_guard_files_pass_without_a_mutant(guards):
    r = baseline(guards)
    assert r.returncode == 0, f"guard files {guards} fail with no mutant (rc {r.returncode})\n{(r.stdout + r.stderr)[-2000:]}"


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(im.MUTANTS))
def test_disabling_the_interlock_fails_a_guard_test(name):
    base = baseline(im.MUTANTS[name].guards)
    assert base.returncode == 0, f"guards of {name!r} fail without the mutant; a failure under it proves nothing"
    r = run_guards(name)
    out = r.stdout + r.stderr
    assert f"[mutant] {name}:" in out, out[-2000:]                # the mutant really was installed
    assert "ERROR collecting" not in out and "INTERNALERROR" not in out, out[-2000:]
    assert "FAILED " in out, out[-2000:]                          # a test assertion failed (not an error)
    assert r.returncode == 1, (f"interlock {name!r} ({im.MUTANTS[name].what}) is unguarded: its guard tests "
                               f"{im.MUTANTS[name].guards} pass with it disabled (rc {r.returncode})\n{out[-2000:]}")
