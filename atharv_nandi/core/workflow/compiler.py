"""``nandi.workflow/1`` YAML -> workflow IR (WP34; WORKFLOWS-SPEC A.2, A.6.6).

py_trees has no YAML loader, so we own this step. ``compile_doc`` turns a parsed YAML document into a tree of
``Node`` objects: defaults applied (sequence memory=true, selector memory=false, primitive default timeouts),
``subtree: {ref: id@version}`` and ``template: <id>`` expanded inline from the ``Library`` (all areas'
``INDEX.yaml``), ``{slot: body}`` filled, ``$name`` arguments substituted, and every node given a stable path
id (``/root/sequence/3/foreach/body``). The IR is pure Python: the validators and the static check run on it
without py_trees; ``engine.py`` maps it onto py_trees behaviours.

Library layout: ``config/workflows/<area>/INDEX.yaml``::

    entries:
    - {id: finaliser_default, version: 1, file: finaliser_default.yaml, sha256: <hex>, primitives_api: 1}

A workflow's identity is ``id@version#sha8`` (``sha8`` = first 8 hex of the file's sha256), recorded in
checkpoints, run manifests and ``TreeRecord.presets[]`` (A.5).
"""
import copy
import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

from atharv_nandi import paths
from . import schema
from .primitives import registry as reg

ROOT = paths.REPO
WORKFLOWS_DIR = os.path.join(ROOT, "config", "workflows")
LIB_AREA = "_lib"
AREA_OWNERS = {"_lib": "WP34", "presets": "WP35", "recovery": "WP35", "missions": "WP35", "rth": "WP38",
               "dock": "WP38", "chores": "WP39", "demos": "WP44"}
FINALISER_DEFAULT = "finaliser_default@1"
RESUME_CHECK = "resume_check@1"
_PARAM_RE = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)$")
_INTERP_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_CALL_RE = re.compile(r"^([a-z_]+)(?:\(([^()]*)\))?$")


class CompileError(ValueError):
    def __init__(self, code: str, path: str, message: str):
        super().__init__(f"{code} at {path}: {message}")
        self.code, self.path, self.message = code, path, message


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_call(text: Any, path: str = "") -> Tuple[str, Optional[str]]:
    """``skip(arc_blocked)`` -> ("skip", "arc_blocked"); ``fail`` -> ("fail", None)."""
    if not isinstance(text, str):
        raise CompileError("bad_value", path, f"{text!r} is not name or name(arg)")
    m = _CALL_RE.match(text.strip())
    if not m:
        raise CompileError("bad_value", path, f"{text!r} is not name or name(arg)")
    arg = m.group(2)
    return m.group(1), (arg.strip() if arg is not None else None)


# ---------------------------------------------------------------------------------------------- IR
@dataclass
class Node:
    type: str
    id: str
    f: Dict[str, Any] = field(default_factory=dict)
    children: List["Node"] = field(default_factory=list)
    label: str = ""
    labels: Dict[str, str] = field(default_factory=dict)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    @property
    def do(self) -> Optional[str]:
        return self.f.get("do") if self.type == "action" else None


@dataclass
class Compiled:
    doc: dict
    ident: str                     # id@version#sha8
    root: Node
    on_pause: Optional[Node]
    on_abort: Optional[Node]
    params: Dict[str, Any]         # name -> default
    param_specs: Dict[str, dict]
    sha256: str

    @property
    def id(self) -> str:
        return self.doc["id"]

    @property
    def version(self) -> int:
        return int(self.doc["version"])

    @property
    def kind(self) -> str:
        return self.doc.get("kind", "")

    def nodes(self):
        for top in (self.root, self.on_pause, self.on_abort):
            if top is not None:
                yield from top.walk()

    def size(self) -> int:
        return sum(1 for _ in self.root.walk())


# ---------------------------------------------------------------------------------------------- library
@dataclass
class Entry:
    id: str
    version: int
    area: str
    path: Optional[str]
    sha256_index: Optional[str]
    primitives_api: Optional[int] = None
    text: Optional[bytes] = None
    _doc: Optional[dict] = None

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    def data(self) -> bytes:
        if self.text is None:
            with open(self.path, "rb") as fh:
                self.text = fh.read()
        return self.text

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.data())

    @property
    def ident(self) -> str:
        return f"{self.ref}#{self.sha256[:8]}"

    def doc(self) -> dict:
        if self._doc is None:
            self._doc = yaml.safe_load(self.data())
        return self._doc


class Library:
    """Every workflow of every area, by ``id@version`` (the validator aggregates the area INDEX files)."""

    def __init__(self, root: Optional[str] = WORKFLOWS_DIR, areas: Optional[List[str]] = None):
        self.root = root
        self.entries: Dict[Tuple[str, int], Entry] = {}
        self.index_issues: List[schema.Issue] = []
        if root and os.path.isdir(root):
            for area in sorted(areas or os.listdir(root)):
                if os.path.isfile(os.path.join(root, area, "INDEX.yaml")):
                    self._load_area(area)

    def _load_area(self, area: str):
        ipath = os.path.join(self.root, area, "INDEX.yaml")
        with open(ipath, encoding="utf-8") as fh:
            idx = yaml.safe_load(fh) or {}
        for i, e in enumerate(idx.get("entries", []) or []):
            where = f"{area}/INDEX.yaml/entries/{i}"
            if not isinstance(e, dict) or not {"id", "version", "file"} <= set(e):
                self.index_issues.append(schema.Issue("bad_index", where, "entry needs id, version, file"))
                continue
            key = (str(e["id"]), int(e["version"]))
            if key in self.entries:
                self.index_issues.append(schema.Issue("duplicate_entry", where, f"{key[0]}@{key[1]} listed twice"))
                continue
            self.entries[key] = Entry(key[0], key[1], area, os.path.join(self.root, area, e["file"]),
                                      e.get("sha256"), e.get("primitives_api"))

    def add(self, doc_or_text: Any, area: str = "test") -> Entry:
        """Register an in-memory workflow (tests, fault matrix); text keeps its sha256 identity."""
        text = doc_or_text if isinstance(doc_or_text, (bytes, str)) else yaml.safe_dump(doc_or_text, sort_keys=False)
        data = text.encode() if isinstance(text, str) else text
        doc = yaml.safe_load(data)
        e = Entry(str(doc["id"]), int(doc["version"]), area, None, None, doc.get("primitives_api"), data, doc)
        self.entries[(e.id, e.version)] = e
        return e

    def get(self, ref: str) -> Entry:
        wid, _, ver = str(ref).partition("@")
        if not ver:
            return self.latest(wid)
        try:
            return self.entries[(wid, int(ver))]
        except (KeyError, ValueError):
            raise KeyError(f"no workflow {ref!r} in the library") from None

    def latest(self, wid: str) -> Entry:
        vs = [e for (i, _), e in self.entries.items() if i == wid]
        if not vs:
            raise KeyError(f"no workflow {wid!r} in the library")
        return max(vs, key=lambda e: e.version)

    def area(self, area: str) -> List[Entry]:
        return [e for e in self.entries.values() if e.area == area]


_DEFAULT_LIB: Optional[Library] = None


def default_library(refresh: bool = False) -> Library:
    global _DEFAULT_LIB
    if refresh or _DEFAULT_LIB is None:
        _DEFAULT_LIB = Library()
    return _DEFAULT_LIB


# ---------------------------------------------------------------------------------------------- expansion
def substitute(obj: Any, args: Dict[str, Any], slots: Dict[str, Any]) -> Any:
    """``$x`` (whole string, keeps the type) and ``${x}`` (in a string) from ``args``; ``{slot: name}`` from slots."""
    if isinstance(obj, dict):
        if set(obj) == {"slot"} and isinstance(obj["slot"], str):
            if obj["slot"] not in slots:
                raise CompileError("unfilled_slot", "", f"template slot {obj['slot']!r} not provided")
            return copy.deepcopy(slots[obj["slot"]])
        return {k: substitute(v, args, slots) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute(v, args, slots) for v in obj]
    if isinstance(obj, str):
        m = _PARAM_RE.match(obj)
        if m and m.group(1) in args:
            return copy.deepcopy(args[m.group(1)])
        return _INTERP_RE.sub(lambda mm: str(args[mm.group(1)]) if mm.group(1) in args else mm.group(0), obj)
    return obj


class _Ctx:
    def __init__(self, library: Library):
        self.library = library
        self.stack: List[str] = []


def _labels(body) -> Tuple[str, Dict[str, str]]:
    if isinstance(body, dict):
        labels = body.get("labels") or {}
        return str(body.get("label", "") or labels.get("en", "")), dict(labels)
    return "", {}


def _compile(node: Any, path: str, ctx: _Ctx) -> Node:
    ntype, body = schema.node_type(node)
    if ntype is None or ntype not in schema.NODE_TYPES:
        raise CompileError("bad_node", path, f"not a workflow node: {node!r}")
    p = f"{path}/{ntype}"
    label, labels = _labels(body)
    if ntype in ("sequence", "selector", "atomic", "parallel"):
        kids = [_compile(c, f"{p}/{i}", ctx) for i, c in enumerate(schema.children_of(ntype, body))]
        f = {}
        if ntype == "sequence":
            f["memory"] = body.get("memory", True) if isinstance(body, dict) else True
        elif ntype == "selector":
            f["memory"] = body.get("memory", False) if isinstance(body, dict) else False
        elif ntype == "parallel":
            f["policy"] = body.get("policy", "all")
        return Node(ntype, p, f, kids, label, labels)
    if ntype == "action":
        return _action(body, p, label, labels)
    if ntype == "foreach":
        f = {"set": body["set"], "skip_ratio_halt": body.get("skip_ratio_halt")}
        return Node("foreach", p, f, [_compile(body["body"], f"{p}/body", ctx)], label or body["set"], labels)
    if ntype == "recover":
        body_node = _compile(body["body"], f"{p}/body", ctx)
        rec = []
        for i, r in enumerate(body.get("recovery", []) or []):
            rp = f"{p}/recovery/{i}"
            rec.append(_action(r, f"{rp}/action", *_labels(r)) if isinstance(r, dict) and "do" in r
                       else _compile(r, rp, ctx))
        rec_seq = Node("sequence", f"{p}/recovery", {"memory": True}, rec, "recovery")
        then = parse_call(body.get("then", "fail"), f"{p}/then")
        f = {"max_rounds": body.get("max_rounds", 2), "then": then}
        return Node("recover", p, f, [body_node, rec_seq], label, labels)
    if ntype == "guard":
        child = body.get("child", body.get("body"))
        f = {"cond": body["cond"], "on_violation": body["on_violation"], "reason": body.get("reason")}
        return Node("guard", p, f, [_compile(child, f"{p}/child", ctx)], label or f"guard {body['cond']}", labels)
    if ntype == "check":
        f = {"cond": body["cond"], "else": parse_call(body.get("else", "fail"), f"{p}/else"),
             "reason": body.get("reason")}
        return Node("check", p, f, [], label or f"check {body['cond']}", labels)
    if ntype == "wait_for":
        f = {"cond": body["cond"], "timeout_s": body.get("timeout_s"),
             "on_timeout": parse_call(body.get("on_timeout", "fail"), f"{p}/on_timeout"), "reason": body.get("reason")}
        return Node("wait_for", p, f, [], label or f"wait_for {body['cond']}", labels)
    if ntype in ("subtree", "template"):
        if ntype == "template" and isinstance(body, str):
            ref, args, slots = body, {}, {}
        elif ntype == "template":
            ref = body["id"]
            args = {k: v for k, v in body.items() if k not in ("id", "body", "label", "labels")}
            slots = {"body": body["body"]} if "body" in body else {}
        else:
            ref, args, slots = body["ref"], dict(body.get("args") or {}), {}
        return _expand(ntype, ref, args, slots, p, label, labels, ctx)
    raise CompileError("slot_outside_template", p, "slot is only valid inside a _lib template")


def _expand(ntype, ref, args, slots, p, label, labels, ctx: _Ctx) -> Node:
    try:
        entry = ctx.library.get(ref if (ntype == "subtree" or "@" in ref) else ref)
    except KeyError as exc:
        raise CompileError("unresolved_ref", p, str(exc)) from None
    if entry.ref in ctx.stack:
        raise CompileError("subtree_cycle", p, " -> ".join(ctx.stack + [entry.ref]))
    doc = entry.doc()
    if doc.get("kind") != "subtree":
        raise CompileError("bad_ref", p, f"{entry.ref} is kind {doc.get('kind')!r}, not subtree")
    defaults = {k: v.get("default") for k, v in (doc.get("params") or {}).items() if isinstance(v, dict)}
    unknown = set(args) - set(defaults)
    if unknown:
        raise CompileError("bad_template_arg", p, f"{entry.ref} has no params {sorted(unknown)}")
    merged = dict(defaults)
    merged.update(args)
    root_yaml = substitute(doc["root"], merged, slots)
    ctx.stack.append(entry.ref)
    try:
        inner = _compile(root_yaml, f"{p}/{entry.ref}", ctx)
    finally:
        ctx.stack.pop()
    lab, labs = _labels(doc)
    return Node("subtree", p, {"ref": entry.ref, "ident": entry.ident, "template": ntype == "template"},
                [inner], label or lab or entry.id, labels or labs)


def _action(body: dict, p: str, label: str, labels: dict) -> Node:
    do = body["do"]
    spec = reg.PRIMITIVES.get(do)
    retry = body.get("retry") or {}
    res = body.get("resumable")
    if res is None:
        res = spec.resumable if spec else "restart"
    kind, to = parse_call(res, f"{p}/resumable")
    if kind == "rollback" and not to:
        to = "arm_park"
    timeout = body.get("timeout_s")
    f = {
        "do": do, "args": dict(body.get("args") or {}),
        "timeout_s": timeout, "timeout_declared": timeout is not None,
        "retry": {"n": retry.get("n", 0), "backoff_s": retry.get("backoff_s", 0.0), "vary": retry.get("vary") or {}},
        "retry_declared": bool(body.get("retry")),
        "resumable": (kind, to), "reason_on_fail": body.get("reason_on_fail"),
    }
    if timeout is None and spec is not None and spec.running:
        f["timeout_s"] = spec.default_timeout_s
    return Node("action", p, f, [], label or do, labels)


def finaliser_ref(doc: dict, key: str) -> Any:
    v = doc.get(key)
    return v if v is not None else {"ref": FINALISER_DEFAULT}


def _top(doc_node: Any, path: str, ctx: _Ctx) -> Node:
    if isinstance(doc_node, dict) and "ref" in doc_node and set(doc_node) <= {"ref", "args"}:
        return _expand("subtree", doc_node["ref"], dict(doc_node.get("args") or {}), {}, path, "", {}, ctx)
    return _compile(doc_node, path, ctx)


def compile_doc(doc: dict, library: Optional[Library] = None, text: Optional[bytes] = None,
                finalisers: bool = True) -> Compiled:
    """Compile one parsed document. ``text`` (the file bytes) gives the sha8; else the canonical YAML dump."""
    library = library or default_library()
    if text is None:
        text = yaml.safe_dump(doc, sort_keys=True).encode()
    sha = sha256_bytes(text if isinstance(text, bytes) else text.encode())
    ctx = _Ctx(library)
    ctx.stack.append(f"{doc.get('id')}@{doc.get('version')}")
    root_yaml = doc["root"]
    if doc.get("slots") or _has_params_refs(doc):
        # a template validated on its own: params at their defaults, every slot a neutral placeholder
        defaults = {k: v.get("default") for k, v in (doc.get("params") or {}).items() if isinstance(v, dict)}
        root_yaml = substitute(root_yaml, defaults, {s: SLOT_PLACEHOLDER for s in doc.get("slots") or ()})
    root = _compile(root_yaml, "/root", ctx)
    on_pause = on_abort = None
    if finalisers and doc.get("kind") != "subtree":
        on_pause = _top(finaliser_ref(doc, "on_pause"), "/on_pause", ctx)
        on_abort = _top(finaliser_ref(doc, "on_abort"), "/on_abort", ctx)
    specs = {k: v for k, v in (doc.get("params") or {}).items() if isinstance(v, dict)}
    return Compiled(doc, f"{doc['id']}@{doc['version']}#{sha[:8]}", root, on_pause, on_abort,
                    {k: v.get("default") for k, v in specs.items()}, specs, sha)


SLOT_PLACEHOLDER = {"check": {"cond": "ready_common", "label": "slot"}}


def _has_params_refs(doc: dict) -> bool:
    return doc.get("kind") == "subtree" and bool(doc.get("params"))


def compile_ref(ref: str, library: Optional[Library] = None) -> Compiled:
    library = library or default_library()
    e = library.get(ref)
    return compile_doc(e.doc(), library, e.data())


def compile_text(text, library: Optional[Library] = None) -> Compiled:
    data = text.encode() if isinstance(text, str) else text
    return compile_doc(yaml.safe_load(data), library, data)


def compile_node(node_yaml: Any, library: Optional[Library] = None, path: str = "/x") -> Node:
    """Compile a bare node (``resume_check``, finalisers built by the engine)."""
    return _top(node_yaml, path, _Ctx(library or default_library()))
