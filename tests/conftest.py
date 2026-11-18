import os

# The repo root is on sys.path through pytest.ini (`pythonpath = .`), so tests import `atharv_nandi.*` and
# `tools.*` as packages. Subprocesses a test starts (`python3 -c ...`, `python3 tools/x.py`) get the same root
# through PYTHONPATH.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_pp = os.environ.get("PYTHONPATH", "")
if REPO not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = REPO + (os.pathsep + _pp if _pp else "")


def pytest_configure(config):
    """Interlock mutation check (WPCOV): with ``NANDI_MUTANT=<name>`` install that mutant (tests/interlock_mutants.py)
    before any test module is imported. Only ``tests/test_interlock_mutation.py`` sets it, in a subprocess."""
    name = os.environ.get("NANDI_MUTANT")
    if not name:
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("interlock_mutants",
                                                  os.path.join(os.path.dirname(__file__), "interlock_mutants.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    patched = mod.apply(name)
    print(f"[mutant] {name}: {', '.join(patched)}")
