"""Workflow validator: structure + semantics + static check + versioning (WP34; WORKFLOWS-SPEC A.6).

``validate_doc(doc)`` runs, in order:

1. **structure** (``schema.validate_structure``);
2. **semantics** (this file): primitives and args exist and are in range (``$param`` args are checked through the
   param's min / max, and such a param must declare both: ``param_unbounded``); guard, condition and reason
   ids are in their enums; a timeout on every action whose primitive can return RUNNING and on every
   ``wait_for``; retries bounded (``retry.n <= 3``, none on a commit primitive: ``retry_on_commit``; recover
   ``max_rounds`` in 1..5 and at most 5 rounds per target along any nesting); ``on_pause`` and ``on_abort``
   present; ``interlocks`` include ``common``; subtree refs resolve and are acyclic (compiler); ``primitives_api``
   matches; ``set()`` only under ``/wf/``; no primitive that would clear an e-stop, bypass the arbiter or remove
   the person pause; ``skip(reason)`` only inside a ``foreach``;
3. **static abstract interpretation** (``static_check.check``);
4. **versioning** against the area ``INDEX.yaml`` (``check_index``): an edited file without a version bump is
   refused (``version_not_bumped``), a bumped file without an INDEX update too (``index_stale``).

CLI (in the image): ``python3 -m atharv_nandi.core.workflow.validate [--reindex] [config/workflows]`` validates every area;
``--reindex`` rewrites the sha256 of entries whose version was bumped (and refuses an edit without a bump).
"""
import os
import sys
from typing import Any, Dict, List, Optional

import yaml

from . import blackboard, compiler, reasons, schema, static_check  # noqa: E402
from .primitives import registry as reg  # noqa: E402

Issue = schema.Issue
MAX_RETRY_N = 3
MAX_ROUNDS = 5
ATOMIC_WAIT_MAX_S = 10.0


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_doc(doc: Any, library: Optional[compiler.Library] = None, text: Optional[bytes] = None,
                 static: bool = True) -> List[Issue]:
    issues = schema.validate_structure(doc)
    if schema.errors(issues):
        return issues
    issues += _header(doc)
    lib = library or compiler.default_library()
    try:
        comp = compiler.compile_doc(doc, lib, text)
    except compiler.CompileError as exc:
        return issues + [Issue(exc.code, exc.path, exc.message)]
    issues += semantic_nodes(comp)
    if static and not schema.errors(issues):
        issues += static_check.check(comp)
    return issues


def validate_text(text, library: Optional[compiler.Library] = None) -> List[Issue]:
    data = text.encode() if isinstance(text, str) else text
    try:
        doc = yaml.safe_load(data)
    except yaml.YAMLError as exc:
        return [Issue("yaml_error", "/", str(exc).splitlines()[0])]
    return validate_doc(doc, library, data)


def _header(doc: dict) -> List[Issue]:
    out = []
    if doc.get("primitives_api") != reg.PRIMITIVES_API:
        out.append(Issue("primitives_api_mismatch", "/primitives_api",
                         f"primitives_api {doc.get('primitives_api')} != {reg.PRIMITIVES_API}"))
    kind = doc.get("kind")
    if kind != "subtree":
        for k in ("on_pause", "on_abort"):
            if k not in doc:
                out.append(Issue(f"missing_{k}", f"/{k}", f"{k} is mandatory (default {{ref: "
                                                          f"{compiler.FINALISER_DEFAULT}}})"))
        if "common" not in (doc.get("interlocks") or []):
            out.append(Issue("missing_common_interlock", "/interlocks", "interlocks must include common"))
    for i, c in enumerate(doc.get("preconditions") or []):
        if c not in reg.CONDITIONS:
            out.append(Issue("unknown_condition", f"/preconditions/{i}", f"unknown Ready-check id {c!r}"))
    for i, c in enumerate(doc.get("interlocks") or []):
        if c not in reg.INTERLOCKS:
            out.append(Issue("unknown_interlock", f"/interlocks/{i}", f"unknown interlock {c!r}"))
    return out


# ---------------------------------------------------------------------------------------------- semantics
def semantic_nodes(comp: compiler.Compiled) -> List[Issue]:
    out: List[Issue] = []
    params = comp.param_specs
    for top in (comp.root, comp.on_pause, comp.on_abort):
        if top is not None:
            _walk(top, out, params, in_foreach=False, rounds=1, in_atomic=False)
    return out


def _walk(n: compiler.Node, out, params, in_foreach: bool, rounds: int, in_atomic: bool):
    t = n.type
    if t == "action":
        _action(n, out, params, in_foreach)
    elif t == "foreach":
        rounds = 1
        in_foreach = True
        if not isinstance(n.f.get("set"), str) or not n.f["set"]:
            out.append(Issue("bad_value", n.id, "foreach needs a target set name"))
    elif t == "recover":
        mr = n.f.get("max_rounds")
        if not isinstance(mr, int) or isinstance(mr, bool) or not 1 <= mr <= MAX_ROUNDS:
            out.append(Issue("unbounded_retry", f"{n.id}/max_rounds", f"max_rounds must be 1..{MAX_ROUNDS}, got {mr!r}"))
        else:
            rounds *= mr
            if rounds > MAX_ROUNDS:
                out.append(Issue("unbounded_retry", f"{n.id}/max_rounds",
                                 f"{rounds} rounds per target along this nesting (max {MAX_ROUNDS})"))
        kind, arg = n.f["then"]
        if kind not in ("continue", "fail", "skip"):
            out.append(Issue("bad_value", f"{n.id}/then", f"then must be continue | fail | skip(reason), got {kind!r}"))
        elif kind == "skip":
            _reason(arg, f"{n.id}/then", out)
            if not in_foreach:
                out.append(Issue("skip_outside_foreach", f"{n.id}/then", "skip(reason) needs an enclosing foreach"))
    elif t == "guard":
        _cond(n.f["cond"], f"{n.id}/cond", out)
        if n.f["on_violation"] not in reasons.ON_VIOLATION:
            out.append(Issue("bad_on_violation", f"{n.id}/on_violation",
                             f"on_violation must be one of {sorted(reasons.ON_VIOLATION)}"))
        if n.f["on_violation"] == "skip_target" and not in_foreach:
            out.append(Issue("skip_outside_foreach", f"{n.id}/on_violation", "skip_target needs an enclosing foreach"))
        if n.f.get("reason") is not None:
            _reason(n.f["reason"], f"{n.id}/reason", out)
    elif t in ("check", "wait_for"):
        _cond(n.f["cond"], f"{n.id}/cond", out)
        key = "else" if t == "check" else "on_timeout"
        kind, arg = n.f[key]
        if kind not in ("fail", "skip", "assist"):
            out.append(Issue("bad_value", f"{n.id}/{key}", f"{key} must be fail | skip(reason) | assist"))
        elif kind == "skip":
            _reason(arg, f"{n.id}/{key}", out)
            if not in_foreach:
                out.append(Issue("skip_outside_foreach", f"{n.id}/{key}", "skip(reason) needs an enclosing foreach"))
        if n.f.get("reason") is not None:
            _reason(n.f["reason"], f"{n.id}/reason", out)
        if t == "wait_for":
            to = n.f.get("timeout_s")
            if to is None:
                out.append(Issue("missing_timeout", f"{n.id}/timeout_s", "wait_for needs timeout_s"))
            elif not _num(to) or to <= 0:
                out.append(Issue("bad_timeout", f"{n.id}/timeout_s", "timeout_s must be > 0"))
    elif t == "parallel" and n.f.get("policy") not in ("all", "one"):
        out.append(Issue("bad_value", f"{n.id}/policy", "policy must be all or one"))
    for c in n.children:
        _walk(c, out, params, in_foreach, rounds, in_atomic or t == "atomic")


def _cond(cid, path, out):
    if cid not in reg.CONDITIONS:
        out.append(Issue("unknown_condition", path, f"unknown condition {cid!r}"))


def _reason(code, path, out):
    if not reasons.is_reason(code):
        out.append(Issue("unknown_reason", path, f"reason {code!r} is not in atharv_nandi/core/workflow/reasons.py"))


def _action(n: compiler.Node, out, params, in_foreach):
    do = n.f["do"]
    if do in reg.FORBIDDEN:
        code = "estop_clear_attempt" if "estop" in do else "forbidden_primitive"
        out.append(Issue(code, n.id, f"{do!r} {reg.FORBIDDEN[do]}: never from YAML"))
        return
    spec = reg.PRIMITIVES.get(do)
    if spec is None:
        out.append(Issue("unknown_primitive", n.id, f"unknown primitive {do!r}"))
        return
    args = n.f["args"]
    for a, v in args.items():
        if a not in spec.args:
            out.append(Issue("unknown_arg", f"{n.id}/args/{a}", f"{do} has no arg {a!r}"))
            continue
        _arg(spec.args[a], v, f"{n.id}/args/{a}", out, params)
    for a, s in spec.args.items():
        if s.required and a not in args:
            out.append(Issue("missing_arg", f"{n.id}/args/{a}", f"{do} needs arg {a!r}"))
    if do == "set":
        _set_key(args.get("key"), f"{n.id}/args/key", out)
    if do in ("skip",) and not in_foreach:
        out.append(Issue("skip_outside_foreach", n.id, "skip(reason) needs an enclosing foreach"))
    if spec.running and not n.f.get("timeout_declared"):
        out.append(Issue("missing_timeout", f"{n.id}/timeout_s", f"{do} can return RUNNING: timeout_s is mandatory"))
    to = n.f.get("timeout_s")
    if n.f.get("timeout_declared") and (not _num(to) or to <= 0 or to > spec.max_timeout_s):
        out.append(Issue("bad_timeout", f"{n.id}/timeout_s", f"timeout_s must be in (0, {spec.max_timeout_s}]"))
    r = n.f["retry"]
    rn = r.get("n")
    if n.f.get("retry_declared") and (not isinstance(rn, int) or isinstance(rn, bool) or not 0 <= rn <= MAX_RETRY_N):
        out.append(Issue("unbounded_retry", f"{n.id}/retry/n", f"retry.n must be 0..{MAX_RETRY_N}, got {rn!r}"))
    if spec.commit and n.f.get("retry_declared") and isinstance(rn, int) and rn > 0:
        out.append(Issue("retry_on_commit", f"{n.id}/retry", f"{do} is a commit point: a retry could fire it "
                                                             "again with a detached item held; recover outside "
                                                             "the atomic instead"))
    if not _num(r.get("backoff_s", 0)) or r.get("backoff_s", 0) < 0:
        out.append(Issue("bad_value", f"{n.id}/retry/backoff_s", "backoff_s must be >= 0"))
    for a, deltas in (r.get("vary") or {}).items():
        if a not in spec.args:
            out.append(Issue("unknown_arg", f"{n.id}/retry/vary/{a}", f"{do} has no arg {a!r} to vary"))
        elif not isinstance(deltas, list) or not deltas or not all(_num(d) for d in deltas):
            out.append(Issue("bad_value", f"{n.id}/retry/vary/{a}", "vary is a non-empty list of numbers"))
    if n.f.get("reason_on_fail") is not None:
        _reason(n.f["reason_on_fail"], f"{n.id}/reason_on_fail", out)
    kind, to_prim = n.f["resumable"]
    if kind not in reg.RESUMABLE:
        out.append(Issue("bad_value", f"{n.id}/resumable", "resumable is restart | continue | rollback(to)"))
    elif kind == "rollback":
        tspec = reg.PRIMITIVES.get(to_prim)
        if tspec is None or not tspec.safe_direction:
            out.append(Issue("bad_value", f"{n.id}/resumable", f"rollback target {to_prim!r} is not a safe primitive"))


def _arg(s: reg.ArgSpec, v, path, out, params):
    if isinstance(v, str) and v.startswith("$") and s.kind not in ("key",):
        name = v[1:]
        p = params.get(name)
        if p is None:
            out.append(Issue("unknown_param", path, f"no param {name!r}"))
            return
        if s.kind in ("num", "int"):
            missing = [b for b in ("min", "max") if not _num(p.get(b))]
            if missing:
                # the static check and the console range need both bounds: a default alone is not a worst case
                out.append(Issue("param_unbounded", path, f"param {name!r} feeds a numeric arg: it needs "
                                                         f"{' and '.join(missing)} (A.7: params within min / max)"))
            for bound in ("min", "max", "default"):
                if _num(p.get(bound)):
                    _range(s, p[bound], f"{path} (param {name}.{bound})", out)
        return
    if s.kind in ("num", "int"):
        if not _num(v) or (s.kind == "int" and not isinstance(v, int)):
            out.append(Issue("bad_arg_type", path, f"expected a number, got {v!r}"))
            return
        _range(s, v, path, out)
    elif s.kind == "enum" and v not in s.choices:
        out.append(Issue("bad_arg_value", path, f"{v!r} not in {list(s.choices)}"))
    elif s.kind == "str" and not isinstance(v, str):
        out.append(Issue("bad_arg_type", path, f"expected a string, got {v!r}"))
    elif s.kind == "bool" and not isinstance(v, bool):
        out.append(Issue("bad_arg_type", path, f"expected a bool, got {v!r}"))
    elif s.kind == "list" and not isinstance(v, list):
        out.append(Issue("bad_arg_type", path, f"expected a list, got {v!r}"))
    elif s.kind == "reason":
        _reason(v, path, out)


def _range(s: reg.ArgSpec, v, path, out):
    if s.min is not None and v < s.min:
        out.append(Issue("out_of_range", path, f"{v} < min {s.min}"))
    if s.max is not None and v > s.max:
        out.append(Issue("out_of_range", path, f"{v} > max {s.max}"))


def _set_key(key, path, out):
    ok, why = blackboard.yaml_key(key)
    if ok:
        return
    out.append(Issue("set_outside_wf", path, f"set() refused: {why}"))
    if isinstance(key, str) and key.startswith("/safety/") and "estop" in key:
        out.append(Issue("estop_clear_attempt", path, "a YAML set() on the e-stop mirror would clear the e-stop"))


# ---------------------------------------------------------------------------------------------- versioning
def check_index(entry: compiler.Entry) -> List[Issue]:
    out = []
    where = f"{entry.area}/{os.path.basename(entry.path or entry.id)}"
    try:
        doc = entry.doc()
    except (OSError, yaml.YAMLError) as exc:
        return [Issue("unreadable", where, str(exc).splitlines()[0])]
    if not isinstance(doc, dict):
        return [Issue("not_a_mapping", where, "not a workflow mapping")]
    if doc.get("id") != entry.id:
        out.append(Issue("index_mismatch", where, f"INDEX id {entry.id!r} != file id {doc.get('id')!r}"))
    if doc.get("version") != entry.version:
        out.append(Issue("index_stale", where, f"file version {doc.get('version')} != INDEX version {entry.version}: "
                                               "update INDEX.yaml (validate.py --reindex)"))
    elif entry.sha256_index is not None and entry.sha256_index != entry.sha256:
        out.append(Issue("version_not_bumped", where, f"{entry.ref} changed (sha256 {entry.sha256[:8]} != INDEX "
                                                      f"{entry.sha256_index[:8]}) without a version bump"))
    if entry.sha256_index is None and entry.path is not None:
        out.append(Issue("index_stale", where, "INDEX entry has no sha256"))
    if entry.primitives_api is not None and entry.primitives_api != doc.get("primitives_api"):
        out.append(Issue("index_mismatch", where, "INDEX primitives_api differs from the file"))
    return out


def validate_library(library: Optional[compiler.Library] = None) -> Dict[str, List[Issue]]:
    """Every INDEX entry of every area: versioning + full validation. Keys are ``area/id@version``."""
    lib = library or compiler.Library()
    out: Dict[str, List[Issue]] = {}
    if lib.index_issues:
        out["INDEX"] = list(lib.index_issues)
    for e in sorted(lib.entries.values(), key=lambda e: (e.area, e.id, e.version)):
        issues = check_index(e)
        if not schema.errors(issues):
            issues += validate_doc(e.doc(), lib, e.data())
        out[f"{e.area}/{e.ref}"] = issues
    return out


def reindex(area_dir: str) -> List[str]:
    """Rewrite the sha256 of each entry whose file version is ahead of the INDEX (bumped); refuse an edit
    without a bump. Returns the refs written."""
    ipath = os.path.join(area_dir, "INDEX.yaml")
    with open(ipath, encoding="utf-8") as fh:
        idx = yaml.safe_load(fh) or {}
    written = []
    for e in idx.get("entries", []):
        path = os.path.join(area_dir, e["file"])
        with open(path, "rb") as fh:
            data = fh.read()
        doc = yaml.safe_load(data)
        sha = compiler.sha256_bytes(data)
        if e.get("sha256") == sha:
            continue
        if e.get("sha256") is not None and doc.get("version") == e.get("version"):
            raise ValueError(f"{e['id']}@{e['version']} changed without a version bump")
        e.update(version=doc["version"], sha256=sha, primitives_api=doc.get("primitives_api"))
        written.append(f"{e['id']}@{e['version']}")
    if written:
        with open(ipath, "w", encoding="utf-8") as fh:
            yaml.safe_dump(idx, fh, sort_keys=False)
    return written


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    do_reindex = "--reindex" in argv
    argv = [a for a in argv if a != "--reindex"]
    root = argv[0] if argv else compiler.WORKFLOWS_DIR
    if do_reindex:
        for area in sorted(os.listdir(root)):
            if os.path.isfile(os.path.join(root, area, "INDEX.yaml")):
                for ref in reindex(os.path.join(root, area)):
                    print(f"reindexed {area}/{ref}")
    bad = 0
    for key, issues in validate_library(compiler.Library(root)).items():
        errs = schema.errors(issues)
        bad += bool(errs)
        print(f"{'FAIL' if errs else 'ok  '} {key}")
        for i in issues:
            print(f"     {i}")
    return 1 if bad else 0


if __name__ == "__main__":                                       # pragma: no cover
    sys.exit(main())
