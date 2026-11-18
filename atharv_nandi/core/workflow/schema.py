"""Structural validator for ``nandi.workflow/1`` (WP34; WORKFLOWS-SPEC A.2, A.6.1).

Hand-rolled (jsonschema is not in the image): known header fields, known node types, required fields and
field types. Semantics (primitives, ranges, reasons, refs) are ``validate.py``; path properties are
``static_check.py``. Every finding is an ``Issue(code, path, message)``; a file is valid when no issue has
``severity == "error"``.

A node is a one-key mapping ``{<type>: <body>}``:

    sequence | selector | atomic   [node, ...]  or  {children: [...], memory: bool, label, labels}
    parallel                       {policy: all|one, children: [...]}
    foreach                        {set: <name>, body: node, skip_ratio_halt: 0..1}
    recover                        {body: node, recovery: [node | action-body, ...], max_rounds: int, then}
    guard                          {cond, on_violation, child: node, reason}
    action                         {do, args, timeout_s, retry: {n, backoff_s, vary}, resumable, reason_on_fail}
    check                          {cond, else: fail|skip(reason)|assist, reason}
    wait_for                       {cond, timeout_s, on_timeout: fail|skip(reason)|assist, reason}
    subtree                        {ref: id@version, args}
    template                       <id>  or  {id: <id>, body: node, <arg>: value, ...}
    slot                           <name>      (only inside a ``_lib`` template)
"""
from dataclasses import dataclass
from typing import Any, List

SCHEMA_ID = "nandi.workflow/1"
KINDS = ("preset", "chore", "recovery", "mission", "subtree")
NODE_TYPES = ("sequence", "selector", "parallel", "foreach", "recover", "guard", "action", "atomic", "check",
              "wait_for", "subtree", "template", "slot")
HEADER_REQUIRED = ("schema", "id", "version", "kind", "primitives_api", "root")
HEADER_OPTIONAL = ("crops", "season", "trigger", "params", "preconditions", "interlocks", "approach_envelope",
                   "on_complete", "outputs", "kpis", "console", "registries", "observers", "on_pause", "on_abort",
                   "mast", "template", "slots", "description", "label", "labels")
NODE_FIELDS = {
    "sequence": ("children", "memory", "label", "labels"),
    "selector": ("children", "memory", "label", "labels"),
    "atomic": ("children", "label", "labels"),
    "parallel": ("children", "policy", "label", "labels"),
    "foreach": ("set", "body", "skip_ratio_halt", "label", "labels"),
    "recover": ("body", "recovery", "max_rounds", "then", "label", "labels"),
    "guard": ("cond", "on_violation", "child", "body", "reason", "label", "labels"),
    "action": ("do", "args", "timeout_s", "retry", "resumable", "reason_on_fail", "label", "labels"),
    "check": ("cond", "else", "reason", "label", "labels"),
    "wait_for": ("cond", "timeout_s", "on_timeout", "reason", "label", "labels"),
    "subtree": ("ref", "args", "label", "labels"),
}
NODE_REQUIRED = {
    "foreach": ("set", "body"), "recover": ("body",), "guard": ("cond", "on_violation"),
    "action": ("do",), "check": ("cond",), "wait_for": ("cond",), "subtree": ("ref",), "parallel": ("children",),
}
RETRY_FIELDS = ("n", "backoff_s", "vary")
ENVELOPE_FIELDS = ("min_standoff_m", "mast_max_under_skirt", "enter_annulus")
ENTER_ANNULUS = ("never", "stowed_only", "allowed")
ON_COMPLETE = ("rth", "next", "hold")


@dataclass(frozen=True)
class Issue:
    code: str
    path: str
    message: str
    severity: str = "error"

    def __str__(self):
        return f"{self.severity} {self.code} at {self.path}: {self.message}"


def errors(issues: List[Issue]) -> List[Issue]:
    return [i for i in issues if i.severity == "error"]


def node_type(node: Any):
    """(type, body) of a node mapping, or (None, None) when it is not a one-key mapping."""
    if isinstance(node, dict) and len(node) == 1:
        (k, v), = node.items()
        return k, v
    return None, None


def children_of(ntype: str, body: Any) -> list:
    """The child node list of a composite body (list form or ``{children: [...]}``)."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        ch = body.get("children", [])
        return ch if isinstance(ch, list) else []
    return []


def validate_structure(doc: Any, where: str = "") -> List[Issue]:
    out: List[Issue] = []
    if not isinstance(doc, dict):
        return [Issue("not_a_mapping", where or "/", "a workflow file is a YAML mapping")]
    for k in HEADER_REQUIRED:
        if k not in doc:
            out.append(Issue("missing_field", f"{where}/{k}", f"header field {k!r} is required"))
    for k in doc:
        if k not in HEADER_REQUIRED and k not in HEADER_OPTIONAL:
            out.append(Issue("unknown_field", f"{where}/{k}", f"unknown header field {k!r}"))
    if "schema" in doc and doc["schema"] != SCHEMA_ID:
        out.append(Issue("bad_schema", f"{where}/schema", f"schema must be {SCHEMA_ID!r}"))
    if "id" in doc and (not isinstance(doc["id"], str) or not doc["id"]):
        out.append(Issue("bad_type", f"{where}/id", "id must be a non-empty string"))
    for k in ("version", "primitives_api"):
        if k in doc and (not isinstance(doc[k], int) or isinstance(doc[k], bool) or doc[k] < 1):
            out.append(Issue("bad_type", f"{where}/{k}", f"{k} must be an int >= 1"))
    if "kind" in doc and doc["kind"] not in KINDS:
        out.append(Issue("bad_value", f"{where}/kind", f"kind must be one of {KINDS}"))
    for k in ("preconditions", "interlocks", "crops", "outputs", "kpis", "registries", "observers", "slots"):
        if k in doc and not isinstance(doc[k], list):
            out.append(Issue("bad_type", f"{where}/{k}", f"{k} must be a list"))
    if "params" in doc:
        out += _params(doc["params"], f"{where}/params")
    if "approach_envelope" in doc:
        env = doc["approach_envelope"]
        if not isinstance(env, dict):
            out.append(Issue("bad_type", f"{where}/approach_envelope", "approach_envelope must be a mapping"))
        else:
            for k in env:
                if k not in ENVELOPE_FIELDS:
                    out.append(Issue("unknown_field", f"{where}/approach_envelope/{k}", f"unknown field {k!r}"))
            if "enter_annulus" in env and env["enter_annulus"] not in ENTER_ANNULUS:
                out.append(Issue("bad_value", f"{where}/approach_envelope/enter_annulus",
                                 f"enter_annulus must be one of {ENTER_ANNULUS}"))
            if "min_standoff_m" in env and not _num(env["min_standoff_m"]):
                out.append(Issue("bad_type", f"{where}/approach_envelope/min_standoff_m", "must be a number"))
    if "on_complete" in doc and doc["on_complete"] not in ON_COMPLETE:
        out.append(Issue("bad_value", f"{where}/on_complete", f"on_complete must be one of {ON_COMPLETE}"))
    if "mast" in doc and doc["mast"] not in ("stowed_only", "any"):
        out.append(Issue("bad_value", f"{where}/mast", "mast must be stowed_only or any"))
    if "trigger" in doc:
        t = doc["trigger"]
        if not (t == "manual" or (isinstance(t, str) and (t.startswith("schedule(") or t.startswith("event(")))
                or (isinstance(t, dict) and set(t) <= {"schedule", "event", "manual"})):
            out.append(Issue("bad_value", f"{where}/trigger", "trigger is manual | schedule(cron) | event(expr)"))
    if "root" in doc:
        out += validate_node(doc["root"], f"{where}/root")
    for k in ("on_pause", "on_abort"):
        if k in doc:
            out += validate_node(doc[k], f"{where}/{k}", allow_ref=True)
    return out


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _pref(v) -> bool:
    """A template param reference (``$name``), substituted before semantics run."""
    return isinstance(v, str) and v.startswith("$")


def _params(params, where) -> List[Issue]:
    out = []
    if not isinstance(params, dict):
        return [Issue("bad_type", where, "params must be a mapping name -> {default, min, max, unit}")]
    for name, p in params.items():
        if not isinstance(p, dict):
            out.append(Issue("bad_type", f"{where}/{name}", "a param is a mapping"))
            continue
        for k in p:
            if k not in ("default", "min", "max", "unit", "enum", "enabled_default", "solved_per_tree", "label"):
                out.append(Issue("unknown_field", f"{where}/{name}/{k}", f"unknown param field {k!r}"))
        if "min" in p and "max" in p and _num(p["min"]) and _num(p["max"]) and p["min"] > p["max"]:
            out.append(Issue("bad_value", f"{where}/{name}", "min > max"))
        d = p.get("default")
        if _num(d) and _num(p.get("min")) and d < p["min"]:
            out.append(Issue("out_of_range", f"{where}/{name}/default", "default below min"))
        if _num(d) and _num(p.get("max")) and d > p["max"]:
            out.append(Issue("out_of_range", f"{where}/{name}/default", "default above max"))
    return out


def validate_node(node: Any, path: str, allow_ref: bool = False) -> List[Issue]:
    out: List[Issue] = []
    if allow_ref and isinstance(node, dict) and set(node) <= {"ref", "args"} and "ref" in node:
        if not isinstance(node["ref"], str) or "@" not in node["ref"]:
            out.append(Issue("bad_ref", f"{path}/ref", "ref must be id@version"))
        return out
    ntype, body = node_type(node)
    if ntype is None:
        return [Issue("bad_node", path, "a node is a one-key mapping {type: body}")]
    if ntype not in NODE_TYPES:
        return [Issue("unknown_node_type", path, f"unknown node type {ntype!r}")]
    p = f"{path}/{ntype}"
    if ntype in ("sequence", "selector", "atomic", "parallel"):
        if ntype == "parallel" and not isinstance(body, dict):
            return [Issue("bad_type", p, "parallel is {policy, children}")]
        if not isinstance(body, (list, dict)):
            return [Issue("bad_type", p, f"{ntype} is a list of nodes or {{children: [...]}}")]
        if isinstance(body, dict):
            out += _fields(ntype, body, p)
            if "children" not in body:
                out.append(Issue("missing_field", f"{p}/children", "children is required"))
            elif not isinstance(body["children"], list):
                out.append(Issue("bad_type", f"{p}/children", "children must be a list"))
            if "memory" in body and not isinstance(body["memory"], bool):
                out.append(Issue("bad_type", f"{p}/memory", "memory must be a bool"))
            if ntype == "parallel" and body.get("policy", "all") not in ("all", "one"):
                out.append(Issue("bad_value", f"{p}/policy", "policy must be all or one"))
        kids = children_of(ntype, body)
        if not kids:
            out.append(Issue("empty_composite", p, f"{ntype} has no children"))
        for i, ch in enumerate(kids):
            out += validate_node(ch, f"{p}/{i}")
        return out
    if ntype == "template":
        if isinstance(body, str):
            return out
        if not isinstance(body, dict) or not isinstance(body.get("id"), str):
            return [Issue("bad_type", p, "template is <id> or {id: <id>, ...args}")]
        if "body" in body:
            out += validate_node(body["body"], f"{p}/body")
        return out
    if ntype == "slot":
        if not isinstance(body, str):
            out.append(Issue("bad_type", p, "slot is a name"))
        return out
    if not isinstance(body, dict):
        return [Issue("bad_type", p, f"{ntype} body must be a mapping")]
    out += _fields(ntype, body, p)
    for k in NODE_REQUIRED.get(ntype, ()):
        if k not in body:
            out.append(Issue("missing_field", f"{p}/{k}", f"{ntype}.{k} is required"))
    if ntype == "action":
        out += _action_fields(body, p)
    elif ntype == "foreach":
        if "set" in body and not isinstance(body["set"], str):
            out.append(Issue("bad_type", f"{p}/set", "set is a target-set name"))
        if "skip_ratio_halt" in body and not _pref(body["skip_ratio_halt"]) and not (_num(body["skip_ratio_halt"]) and 0 < body["skip_ratio_halt"] <= 1):
            out.append(Issue("bad_value", f"{p}/skip_ratio_halt", "skip_ratio_halt must be in (0, 1]"))
        if "body" in body:
            out += validate_node(body["body"], f"{p}/body")
    elif ntype == "recover":
        if "body" in body:
            out += validate_node(body["body"], f"{p}/body")
        rec = body.get("recovery", [])
        if not isinstance(rec, list):
            out.append(Issue("bad_type", f"{p}/recovery", "recovery must be a list"))
            rec = []
        for i, r in enumerate(rec):
            if isinstance(r, dict) and "do" in r:
                out += _fields("action", r, f"{p}/recovery/{i}") + _action_fields(r, f"{p}/recovery/{i}")
            else:
                out += validate_node(r, f"{p}/recovery/{i}")
        if "max_rounds" in body and not _pref(body["max_rounds"]) and (not isinstance(body["max_rounds"], int) or isinstance(body["max_rounds"], bool)):
            out.append(Issue("bad_type", f"{p}/max_rounds", "max_rounds must be an int"))
        if "then" in body and not isinstance(body["then"], str):
            out.append(Issue("bad_type", f"{p}/then", "then is continue | fail | skip(reason)"))
    elif ntype == "guard":
        child = body.get("child", body.get("body"))
        if child is None:
            out.append(Issue("missing_field", f"{p}/child", "guard.child is required"))
        else:
            out += validate_node(child, f"{p}/child")
    elif ntype in ("wait_for",):
        if "timeout_s" in body and not _num(body["timeout_s"]):
            out.append(Issue("bad_type", f"{p}/timeout_s", "timeout_s must be a number"))
    elif ntype == "subtree":
        if "ref" in body and (not isinstance(body["ref"], str) or "@" not in body["ref"]):
            out.append(Issue("bad_ref", f"{p}/ref", "ref must be id@version"))
    return out


def _fields(ntype, body, p) -> List[Issue]:
    allowed = NODE_FIELDS.get(ntype, ())
    return [Issue("unknown_field", f"{p}/{k}", f"unknown {ntype} field {k!r}") for k in body if k not in allowed]


def _action_fields(body, p) -> List[Issue]:
    out = []
    if "do" in body and not isinstance(body["do"], str):
        out.append(Issue("bad_type", f"{p}/do", "do is a primitive name"))
    if "args" in body and not isinstance(body["args"], dict):
        out.append(Issue("bad_type", f"{p}/args", "args must be a mapping"))
    if "timeout_s" in body and not _num(body["timeout_s"]):
        out.append(Issue("bad_type", f"{p}/timeout_s", "timeout_s must be a number"))
    if "retry" in body:
        r = body["retry"]
        if not isinstance(r, dict):
            out.append(Issue("bad_type", f"{p}/retry", "retry is {n, backoff_s, vary}"))
        else:
            for k in r:
                if k not in RETRY_FIELDS:
                    out.append(Issue("unknown_field", f"{p}/retry/{k}", f"unknown retry field {k!r}"))
            if "vary" in r and not isinstance(r["vary"], dict):
                out.append(Issue("bad_type", f"{p}/retry/vary", "vary is {arg: [deltas]}"))
    return out
