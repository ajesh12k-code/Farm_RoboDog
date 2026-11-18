"""WP9 data products (pure Python, no ROS, no MuJoCo): TreeRecord ``nandi.tree/1``, MarkRecord ``nandi.mark/1``
checks, HarvestRecord ``nandi.harvest/2``, the uplink contract checks and the kis/v1 ``tele/tree``, ``tele/mark``,
``tele/mission``, ``tele/harvest``, ``tele/kpi`` and ``tele/alert`` mapping (all PROPOSED streams).

Spec: ORCHARD-MVP-SPEC 1.3 (O-15, O-16, O-22, O-23), 5.4 (kis/v1 additions), 5.6 (data products, leads stay leads,
personal data); MISSION-PRESETS 0.6 (record shapes), 0.5 (reason codes); MVP-SPEC 5.9 (uplink rules).

Rules enforced here (each one has a test in tests/test_records.py or tests/test_uplink_contract.py):
    size        a TreeRecord is <= 16 KiB as compact JSON (``TREE_RECORD_MAX_BYTES``); an oversize record is trimmed
                in a fixed order (keyframe paths, symptom rows beyond 40, size histogram) and labelled ``trimmed``,
                then refused if still too large
    no images   no key that names image data and no value that is bytes, an array buffer, a ``data:`` URI or a long
                base64-looking string, anywhere in a record or uplink payload
    no persons  no key matching ``face|name|phone|email|person_*|worker_*|holder|tag_id|track`` (MVP-SPEC 5.9 plus
                the WP29 tag / track words); person-related alerts carry no coordinates at all and leave the robot
                only as the hourly ``person_hold_count`` aggregate (OS 5.6, OQ-31 A)
    leads       every ``symptoms[]`` row has ``stage: lead`` (only the hub writes ``confirmed``); no uplink payload has
                a ``diagnosis`` / ``treatment`` / ``spray`` / ``advice`` / ``finding(s)`` key: the WP26 leaf block's
                ``findings[]`` leaves the robot only as the count ``leads_n``
    next        ``next_actions[]`` only from the enum ``revisit | tree360 | deep_scan | leaf_scan | trunk_scan |
                human_inspect | mark_jobs:<n>`` (never treatment advice)
    tenant      ``sim-*`` tenants only, ``data_provenance: simulated`` everywhere
"""
from __future__ import annotations

import copy
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from atharv_nandi.core import kis_mapper as km

# ---------------------------------------------------------------------- schema ids and limits
TREE_SCHEMA = "nandi.tree/1"
MARK_SCHEMA = "nandi.mark/1"
HARVEST_SCHEMA = "nandi.harvest/2"
MISSION_SCHEMA = "nandi.tour/1"
TENANT = "sim-nandi"
PROVENANCE = "simulated"
TREE_RECORD_MAX_BYTES = 16 * 1024
NEXT_ACTIONS = ("revisit", "tree360", "deep_scan", "leaf_scan", "trunk_scan", "human_inspect")
MARK_JOBS_RE = re.compile(r"^mark_jobs:[0-9]{1,4}$")
MARK_KINDS = ("fruit", "prune", "rootstock_sprout", "skirt_lift", "malformed_panicle", "sanitation_deleaf", "sucker",
              "bunch")
TOOL_HINTS = ("pole_picker", "lopper", "sickle", "saw", "human_cut")
H_CLASSES = ("neeche", "beech", "upar")
SIDES = ("N", "E", "S", "W")
PRESETS_SEEN = ("scan", "tree360", "leaf_scan", "trunk_scan", "fruit_scout", "harvest", "prune", "mark",
                "dock_unload")
SYMPTOM_ROWS_MAX = 40          # trim order step 2 (a tree with more distinct leads is summarised by `units`)

# MP 0.5 reason codes + the WP8 / WP27 extensions a MarkRecord may carry
MARK_REASONS = ("out_of_reach_marked", "paste_required", "twist_failed", "no_clearance", "bagged", "over_capacity",
                "human_cut", "not_in_mvp", "leaf_minimum", "keep_wood", "operator", "sanitation", "desucker",
                "bunch")

# ---------------------------------------------------------------------- uplink contract words
# Personal keys are matched on the key's tokens (split at "_", "-", ".", camelCase), not on substrings: a person
# word anywhere in the key (``people_n``, ``nearby_worker_xy``, ``operator_id``), an identity word, a name of a
# person (``name`` alone or with first / last / full / user / display), a face or track identifier. Object names
# (``cultivar_name``, ``filename``), ``tool_holder``, ``tracking_ok`` and ``face_count`` are not personal.
PERSONAL_TOKENS = {"person", "persons", "people", "human", "humans", "worker", "workers", "operator", "operators",
                   "farmer", "farmers", "crew", "owner", "user", "users", "username", "phone", "phones", "mobile",
                   "email", "mail", "aadhaar", "aadhar", "adhaar", "address"}
NAME_PERSON_PREFIX = {"first", "last", "full", "user", "display", "given", "family", "sur", "nick", "middle"}
FACE_ID_TOKENS = {"id", "ids", "embedding", "embeddings", "crop", "crops", "bbox", "box", "boxes", "landmarks",
                  "encoding", "vector", "match"}
TRACK_ID_TOKENS = {"id", "ids"}
_TOKEN_RE = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")
# keys allowed although they contain a personal word: a count of person holds (no identity, no place)
PERSONAL_KEY_ALLOW = {"person_hold_count", "person_holds", "person_pauses"}


def key_tokens(key: str) -> List[str]:
    return [t.lower() for part in re.split(r"[_\-.\s]+", str(key)) for t in _TOKEN_RE.findall(part)]


def personal_key(key: str) -> bool:
    """True if a JSON key names a person or carries personal data (uplink contract, OS 5.6)."""
    kl = str(key).lower()
    if kl in PERSONAL_KEY_ALLOW:
        return False
    toks = key_tokens(key)
    if not toks:
        return False
    if any(t in PERSONAL_TOKENS for t in toks):
        return True
    if kl in ("name", "names", "fullname", "firstname", "lastname", "surname", "tag", "tag_id", "faces", "track",
              "tracks"):
        return True
    for i, t in enumerate(toks):
        prev = toks[i - 1] if i else None
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if t in ("name", "names") and prev in NAME_PERSON_PREFIX:
            return True
        if t in ("face", "faces") and (nxt in FACE_ID_TOKENS or prev in FACE_ID_TOKENS):
            return True
        if t in ("track", "tracks") and (nxt in TRACK_ID_TOKENS):
            return True
    return False
IMAGE_KEY_RE = re.compile(r"(^|_)(image|images|img|jpe?g|png|rgb|rgba_px|pixels|frame_data|thumbnail|thumb|b64|"
                          r"base64|blob|video|mp4)($|_)", re.I)
ADVICE_KEY_RE = re.compile(r"^(diagnosis|diagnoses|treatment|treatments|spray|advice|recommendation|"
                           r"recommendations|finding|findings|confirmed)$", re.I)
COORD_KEYS = {"x", "y", "z", "xy", "xyz", "pos", "position", "pose", "lat", "lon", "lng", "latitude", "longitude",
              "geo", "enu", "e", "n", "u", "bearing", "bearing_deg", "range", "range_m", "world", "point", "at",
              "track", "path", "waypoints", "clock", "side"}
B64_RE = re.compile(r"^[A-Za-z0-9+/=_-]{256,}$")
PERSON_ALERTS = ("person_hold", "orbit_pause", "person_pause", "protective_person", "worker_in_field",
                 "hand_in_envelope", "person_hold_count")

# WP6 / WP7 / WP24 / WP25 / WP28 alert ids -> the kis/v1 tele/alert ids (OS 5.4, MVP-SPEC 5.7)
ALERT_MAP = {
    "orbit_blocked": "orbit_blocked",
    "crown_retract": "crown_retract", "strike": "crown_retract", "wind_retract": "crown_retract",
    "stuck": "stuck", "sink": "stuck",
    "local_link_lost_hold": "local_link_lost_hold",
    "cloud_link_lost_hold": "cloud_link_lost_hold",
    "low_battery_return": "low_battery_return",
    "self_right_failed": "self_right_failed",
    "tool_fault": "tool_fault",
    "obstacle_stop": "obstacle_stop", "protective_stop": "obstacle_stop", "animal_hold": "obstacle_stop",
    "rtk_lost": "rtk_lost", "rtk_return_odometry": "rtk_lost",
    "stability_stop": "stability_stop", "fallen": "stability_stop",
    "geofence": "geofence",
}
UPLINK_ALERT_IDS = tuple(sorted(set(ALERT_MAP.values()) | {"person_hold_count"}))


# ---------------------------------------------------------------------- identity
def mission_id(device_id: str, when: Optional[datetime] = None) -> str:
    """Platform mission id ``{device}-{UTC yyyymmddThhmmZ}``."""
    when = when or datetime.now(timezone.utc)
    return f"{device_id}-{when.astimezone(timezone.utc):%Y%m%dT%H%MZ}"


def identity(run_id: str, device_id: str = "nandi-sim", mission: Optional[str] = None,
             tenant: str = TENANT) -> Dict[str, str]:
    if not km.SIM_TENANT_RE.match(tenant or ""):
        raise ValueError(f"tenant {tenant!r} refused: sim-* tenants only")
    return {"run_id": str(run_id), "mission_id": mission or mission_id(device_id), "device_id": str(device_id),
            "tenant": tenant, "data_provenance": PROVENANCE}


def plain(obj: Any) -> Any:
    """JSON-plain copy: numpy scalars / arrays -> Python numbers / lists, tuples -> lists, non-finite floats -> None
    (WP25 / WP26 fragments carry numpy floats; kis/v1 refuses NaN)."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return plain(obj.tolist())
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def json_size(obj: Any) -> int:
    return len(json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
                          default=lambda o: f"<{type(o).__name__}>").encode("utf-8"))


# ---------------------------------------------------------------------- contract scans
def _walk(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield p, str(k), v
            yield from _walk(v, p)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def _binary_value(v: Any) -> bool:
    if isinstance(v, (bytes, bytearray, memoryview)):
        return True
    if (hasattr(v, "__array_interface__") or hasattr(v, "tobytes")) and getattr(v, "ndim", 1) != 0:
        return True                       # an array buffer (a numpy scalar is a number, not binary)
    if isinstance(v, str) and (v.startswith("data:") or B64_RE.match(v)):
        return True
    return False


def contract_violations(obj: Any, *, person_event: bool = False) -> List[str]:
    """Every uplink-contract breach in ``obj`` as ``"<rule>:<path>"`` ([] = clean). ``person_event`` additionally
    refuses any coordinate-like key (person events carry no place, OS 5.6)."""
    out: List[str] = []
    if _binary_value(obj):
        out.append("binary:")
    for path, key, val in _walk(obj):
        kl = key.lower()
        if personal_key(key):
            out.append(f"personal_key:{path}")
        if IMAGE_KEY_RE.search(kl):
            out.append(f"image_key:{path}")
        if ADVICE_KEY_RE.match(kl):
            out.append(f"advice_key:{path}")
        if _binary_value(val):
            out.append(f"binary:{path}")
        if person_event and kl in COORD_KEYS:
            out.append(f"person_coordinates:{path}")
        if kl == "stage" and "symptoms[" in path and val != "lead":
            out.append(f"lead_not_lead:{path}")
        if kl == "next_actions" and isinstance(val, list):
            for a in val:
                if not valid_next_action(a):
                    out.append(f"next_action:{path}:{a}")
        if kl == "data_provenance" and val != PROVENANCE:
            out.append(f"provenance:{path}")
        if kl == "tenant" and isinstance(val, str) and not km.SIM_TENANT_RE.match(val):
            out.append(f"tenant:{path}")
    return out


def valid_next_action(a: Any) -> bool:
    return isinstance(a, str) and (a in NEXT_ACTIONS or bool(MARK_JOBS_RE.match(a)))


# ---------------------------------------------------------------------- MarkRecord
def mark_key(m: dict) -> str:
    """Stable identity of one MARK (job sheet cross-reference): tree, kind, target, clock and height."""
    return "|".join(str(x) for x in (m.get("tree_id"), m.get("kind"), m.get("target_id") or "",
                                     m.get("clock"), m.get("h_m")))


def validate_mark(m: dict) -> List[str]:
    errs = []
    if not isinstance(m, dict):
        return ["mark: not an object"]
    if m.get("schema") != MARK_SCHEMA:
        errs.append("schema")
    for k in ("tree_id", "kind", "clock", "side", "h_class", "h_m", "reason"):
        if m.get(k) is None:
            errs.append(f"missing:{k}")
    if m.get("kind") not in MARK_KINDS:
        errs.append(f"kind:{m.get('kind')}")
    if m.get("h_class") is not None and m["h_class"] not in H_CLASSES:
        errs.append(f"h_class:{m['h_class']}")
    if m.get("side") is not None and m["side"] not in SIDES:
        errs.append(f"side:{m['side']}")
    c = m.get("clock")
    if c is not None and (not isinstance(c, int) or not 1 <= c <= 12):
        errs.append(f"clock:{c}")
    if m.get("tool_hint") is not None and m["tool_hint"] not in TOOL_HINTS:
        errs.append(f"tool_hint:{m['tool_hint']}")
    d = m.get("diameter_mm")
    if d is not None and bool(m.get("paste_required")) != (float(d) > 12.0):
        errs.append("paste_required")
    errs += contract_violations({k: v for k, v in m.items() if k not in ("clock", "side")})
    return errs


def dedupe_marks(marks: Iterable[dict]) -> List[dict]:
    """One MARK per human job: the same tree / kind / target / clock / height seen twice (SCOUT then HARVEST) is one."""
    out, seen = [], set()
    for m in marks:
        k = mark_key(m)
        if k in seen:
            continue
        seen.add(k)
        out.append(m)
    return out


# ---------------------------------------------------------------------- HarvestRecord v2
HARVEST_OUTCOMES = ("delivered", "marked", "skipped", "dropped", "failed", "halted")
_V1_TO_V2 = {"picked": "delivered", "cut": "delivered", "marked": "marked", "skipped": "skipped",
             "dropped": "dropped", "failed": "failed", "halted": "halted"}


def harvest_record_v2(v1: dict, ident: Dict[str, str], entry: Optional[dict] = None, *, kind: str = "harvest",
                      t: Optional[float] = None, mass_kg: Optional[float] = None, stage: Optional[str] = None,
                      zone_id: Optional[str] = None, assisted: Sequence[str] = ()) -> dict:
    """WP8 ``nandi.harvest/1`` item -> ``nandi.harvest/2``: + identity, location (block, row, physical tag, zone),
    ``kind`` (harvest | prune), the v2 outcome (picked / cut -> delivered), mass, stage and sim time. No geo."""
    entry = entry or {}
    out = {"schema": HARVEST_SCHEMA, **ident,
           "zone_id": zone_id or entry.get("zone_id"), "block_id": entry.get("block_id"),
           "row_id": entry.get("row_id"), "tree_id": v1.get("tree_id") or entry.get("tree_id"),
           "physical_tag": entry.get("physical_tag"), "crop": v1.get("crop") or entry.get("crop"),
           "item_id": v1.get("target_id"), "kind": kind, "method": v1.get("method"),
           "outcome": _V1_TO_V2.get(v1.get("outcome"), "failed"), "outcome_v1": v1.get("outcome"),
           "reason": v1.get("reason") or "", "stage": stage, "pedicel_mm": v1.get("pedicel_mm"),
           "pedicel_error_mm": v1.get("pedicel_error_mm"), "cell": v1.get("cell"), "attempts": v1.get("attempts", 1),
           "cycle_s": v1.get("cycle_s"), "mass_g": None if mass_kg is None else round(float(mass_kg) * 1000.0, 1),
           "t_sim": None if t is None else round(float(t), 2),
           "assisted": sorted(set(v1.get("assisted") or ()) | set(assisted))}
    return out


def validate_harvest(h: dict) -> List[str]:
    errs = []
    if h.get("schema") != HARVEST_SCHEMA:
        errs.append("schema")
    for k in ("run_id", "device_id", "tenant", "data_provenance", "tree_id", "item_id", "crop", "outcome"):
        if h.get(k) in (None, ""):
            errs.append(f"missing:{k}")
    if h.get("outcome") not in HARVEST_OUTCOMES:
        errs.append(f"outcome:{h.get('outcome')}")
    errs += contract_violations(h)
    return errs


# ---------------------------------------------------------------------- TreeRecord
def _r(v: Any, nd: int = 3):
    return round(float(v), nd) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None


class TreeRecordBuilder:
    """Merges the per-preset fragments of one tree visit into ``TreeRecord nandi.tree/1`` (MP 0.6).

    Fragments: SCAN (this module's ``scan_fragment``), WP25 TREE 360 (``/nandi/tree/record``), WP26 (``record_fragment``
    of LEAF / TRUNK SCAN and FRUIT SCOUT), WP8 harvest / prune results and MARKs. A later fragment of the same kind
    replaces the earlier block (the newest visit wins); symptoms are merged and de-duplicated; ``next_actions`` is
    recomputed from the merged state and always ends with ``mark_jobs:<n>`` when the tree has open MARKs."""

    def __init__(self, ident: Dict[str, str], entry: dict, zone_id: Optional[str] = None):
        self.ident = dict(ident)
        self.entry = dict(entry)
        self.zone_id = zone_id or entry.get("zone_id")
        self.presets: List[str] = []
        self.visit_t: Optional[float] = None
        self.blocks: Dict[str, Any] = {}
        self.symptoms: List[dict] = []
        self.marks: List[dict] = []
        self.detectors: set = set()
        self.assisted: set = set()
        self.next_hint: List[str] = []
        self.flags: List[str] = []

    @property
    def tree_id(self) -> str:
        return self.entry["tree_id"]

    def _visit(self, preset: str, t: Optional[float]):
        if preset not in self.presets:
            self.presets.append(preset)
        if t is not None:
            self.visit_t = float(t) if self.visit_t is None else min(self.visit_t, float(t))

    def add_symptoms(self, rows: Iterable[dict]):
        keep = ("class", "band", "clock", "h_class", "h_m", "unit", "count", "score", "stage")
        seen = {(s["class"], s["band"], s["clock"], s["unit"]) for s in self.symptoms}
        for s in rows:
            if s.get("stage") != "lead":
                raise ValueError(f"symptom stage {s.get('stage')!r}: Nandi writes leads only (OS 5.6)")
            k = (s.get("class"), s.get("band"), s.get("clock"), s.get("unit"))
            if k in seen:
                continue
            seen.add(k)
            self.symptoms.append({kk: s.get(kk) for kk in keep})

    def merge_scan(self, frag: dict, t: Optional[float] = None):
        self._visit("scan", t)
        cov = frag.get("coverage")
        if cov and not (self.blocks.get("coverage") or {}).get("mode") in ("full", "two_sided"):
            self.blocks["coverage"] = cov          # a TREE 360 coverage block is never replaced by a SCAN one
        if frag.get("canopy"):
            self.blocks["canopy"] = frag["canopy"]
        self.add_symptoms(frag.get("symptoms") or ())
        self.flags = sorted(set(self.flags) | set(frag.get("flags") or ()))
        self.detectors |= set(frag.get("detector") or ())
        self.assisted |= set(frag.get("assisted") or ())

    def merge_tree360(self, frag: dict, t: Optional[float] = None):
        self._visit("tree360", t if t is not None else frag.get("visit_t"))
        if frag.get("coverage"):
            self.blocks["coverage"] = frag["coverage"]
        if frag.get("media"):
            self.blocks["media"] = frag["media"]
        self.detectors |= set(frag.get("detector") or ())
        self.assisted |= set(frag.get("assisted") or ())
        self.next_hint += [a for a in frag.get("next_actions") or () if valid_next_action(a)]

    def merge_perception(self, frag: dict, preset: str, t: Optional[float] = None):
        """WP26 ``record_fragment`` (LEAF SCAN / TRUNK SCAN / FRUIT SCOUT)."""
        self._visit(preset, t)
        self.add_symptoms(frag.get("symptoms") or ())
        for k in ("trunk", "leaf", "fruit", "units"):
            if frag.get(k) is not None:
                self.blocks[k] = frag[k]
        self.detectors |= set(frag.get("detector") or ())
        self.assisted |= set(frag.get("assisted") or ())

    def merge_harvest(self, results: Sequence[dict], marks: Sequence[dict], kg: Optional[float] = None,
                      t: Optional[float] = None):
        self._visit("harvest", t)
        picked = [r for r in results if r.get("outcome") in ("picked", "delivered")]
        ped = [float(r["pedicel_mm"]) for r in picked if isinstance(r.get("pedicel_mm"), (int, float))]
        h = self.blocks.get("harvest") or {"picked_n": 0, "marked_n": 0, "kg": 0.0, "pedicel_mm_mean": None,
                                           "_ped": []}
        h["picked_n"] += len(picked)
        h["marked_n"] += len(marks)
        h["kg"] = round(float(h["kg"]) + float(kg or 0.0), 3)
        h["_ped"] = list(h.get("_ped") or []) + ped
        h["pedicel_mm_mean"] = round(sum(h["_ped"]) / len(h["_ped"]), 2) if h["_ped"] else None
        self.blocks["harvest"] = h
        self.add_marks(marks)

    def merge_prune(self, results: Sequence[dict], marks: Sequence[dict], t: Optional[float] = None):
        self._visit("prune", t)
        p = self.blocks.get("prune") or {"cuts_n": 0, "marked": []}
        p["cuts_n"] += sum(1 for r in results if r.get("outcome") in ("cut", "delivered"))
        p["marked"] = list(p["marked"]) + [{"kind": m["kind"], "h_class": m.get("h_class"),
                                            "diameter_mm": m.get("diameter_mm"),
                                            "paste_required": bool(m.get("paste_required")),
                                            "tool_hint": m.get("tool_hint")} for m in marks]
        self.blocks["prune"] = p
        self.add_marks(marks)

    def add_marks(self, marks: Iterable[dict]):
        self.marks = dedupe_marks(list(self.marks) + list(marks))

    def set_harvest_kg(self, kg: float):
        h = self.blocks.setdefault("harvest", {"picked_n": 0, "marked_n": 0, "kg": 0.0, "pedicel_mm_mean": None})
        h["kg"] = round(float(kg), 3)

    def next_actions(self) -> List[str]:
        out = [a for a in self.next_hint if not a.startswith("mark_jobs:")]
        cov = self.blocks.get("coverage") or {}
        if cov.get("partial") or cov.get("status") == "partial":
            out.append("revisit")
        did = set(self.presets)
        strong = [s for s in self.symptoms if float(s.get("score") or 0.0) >= 0.5]
        if any(s["unit"] in ("per_panicle", "per_shoot") for s in strong) and "tree360" not in did:
            out.append("tree360")
        if strong and "tree360" in did:
            out.append("deep_scan")
        if any(s["unit"] == "per_leaf" for s in strong) and "leaf_scan" not in did:
            out.append("leaf_scan")
        if any(s["unit"] == "trunk" for s in strong) and "trunk_scan" not in did:
            out.append("trunk_scan")
        if any(s["unit"] in ("trunk", "per_leaf") for s in self.symptoms) and ("leaf_scan" in did or
                                                                              "trunk_scan" in did):
            out.append("human_inspect")
        if self.marks:
            out.append(f"mark_jobs:{len(self.marks)}")
        res = []
        for a in out:
            if a not in res and valid_next_action(a):
                res.append(a)
        return res

    def build(self) -> dict:
        e = self.entry
        geo = e.get("geo") or {}
        rec = {"schema": TREE_SCHEMA, **self.ident,
               "zone_id": self.zone_id, "block_id": e.get("block_id"), "row_id": e.get("row_id"),
               "tree_id": e["tree_id"], "physical_tag": e.get("physical_tag"), "crop": e.get("crop"),
               "cultivar": e.get("cultivar"), "form": e.get("form"),
               "geo": {"frame": geo.get("frame"), "x": _r(geo.get("x")), "y": _r(geo.get("y"))},
               "visit_t": _r(self.visit_t, 2), "presets": list(self.presets), "flags": list(self.flags)}
        for k in ("coverage", "canopy"):
            if k in self.blocks:
                rec[k] = copy.deepcopy(self.blocks[k])
        rec["symptoms"] = [dict(s) for s in self.symptoms]
        for k in ("units", "trunk", "leaf", "fruit"):
            if k in self.blocks:
                rec[k] = copy.deepcopy(self.blocks[k])
        if "harvest" in self.blocks:
            rec["harvest"] = {k: v for k, v in self.blocks["harvest"].items() if not k.startswith("_")}
        if "prune" in self.blocks:
            rec["prune"] = copy.deepcopy(self.blocks["prune"])
        if "media" in self.blocks:
            rec["media"] = copy.deepcopy(self.blocks["media"])
        rec["marks_n"] = len(self.marks)
        rec["detector"] = sorted(self.detectors) or ["simtruth"]
        rec["assisted"] = sorted(self.assisted)
        rec["next_actions"] = self.next_actions()
        return fit_tree_record(plain(rec))


def fit_tree_record(rec: dict, limit: int = TREE_RECORD_MAX_BYTES) -> dict:
    """Trim a TreeRecord to ``limit`` bytes in a fixed order, labelling what was trimmed; raise if impossible."""
    if json_size(rec) <= limit:
        return rec
    rec = copy.deepcopy(rec)
    trimmed = []
    media = rec.get("media") or {}
    if media.get("keyframe_paths"):
        media["keyframes_n"] = len(media["keyframe_paths"])
        media["keyframe_paths"] = media["keyframe_paths"][:4]
        trimmed.append("media.keyframe_paths")
    if json_size(rec) > limit and len(rec.get("symptoms") or []) > SYMPTOM_ROWS_MAX:
        rec["symptoms_n"] = len(rec["symptoms"])
        rec["symptoms"] = sorted(rec["symptoms"], key=lambda s: -float(s.get("score") or 0))[:SYMPTOM_ROWS_MAX]
        trimmed.append("symptoms")
    if json_size(rec) > limit and isinstance(rec.get("fruit"), dict) and "size_mm_hist" in rec["fruit"]:
        rec["fruit"].pop("size_mm_hist")
        trimmed.append("fruit.size_mm_hist")
    if json_size(rec) > limit and isinstance(rec.get("leaf"), dict):
        for k in ("findings", "revisit_leaves"):
            if k in rec["leaf"]:
                rec["leaf"][f"{k}_n"] = len(rec["leaf"].pop(k) or [])
        trimmed.append("leaf")
    rec["trimmed"] = trimmed
    if json_size(rec) > limit:
        raise ValueError(f"TreeRecord {rec.get('tree_id')}: {json_size(rec)} B > {limit} B after trimming")
    return rec


def validate_tree_record(rec: dict) -> List[str]:
    errs = []
    if rec.get("schema") != TREE_SCHEMA:
        errs.append("schema")
    for k in ("run_id", "mission_id", "device_id", "tenant", "data_provenance", "block_id", "row_id", "tree_id",
              "physical_tag", "crop", "presets", "next_actions", "detector", "assisted"):
        if rec.get(k) is None:
            errs.append(f"missing:{k}")
    n = json_size(rec)
    if n > TREE_RECORD_MAX_BYTES:
        errs.append(f"size:{n}")
    for s in rec.get("symptoms") or []:
        if s.get("stage") != "lead":
            errs.append(f"symptom_stage:{s.get('stage')}")
    for a in rec.get("next_actions") or []:
        if not valid_next_action(a):
            errs.append(f"next_action:{a}")
    leaf = rec.get("leaf") or {}
    # the WP26 leaf block's findings[] are leads (labelled colony_symptom_level); allowed on the robot, scanned below
    errs += [v for v in contract_violations({k: v for k, v in rec.items() if k != "leaf"})]
    errs += [v for v in contract_violations({k: v for k, v in leaf.items() if k != "findings"})]
    for f in leaf.get("findings") or []:
        if isinstance(f, dict) and f.get("stage", "lead") != "lead":
            errs.append("leaf_finding_not_lead")
    return errs


# ---------------------------------------------------------------------- SCAN fragment (this package's SCAN step)
def scan_fragment(entry: dict, side: str, bands_seen: Sequence[str], symptoms: Sequence[dict], canopy: dict,
                  flags: Sequence[str], frames: int, detector: Sequence[str] = ("simtruth",),
                  assisted: Sequence[str] = ()) -> dict:
    """TreeRecord pieces of one SCAN pass of one tree: per-side band coverage (``mode: scan``), LiDAR canopy profile,
    canopy- and trunk-band leads, triage ``flags``."""
    return {"coverage": {"mode": "scan", "sides": [side], "bands_by_side": {side: sorted(set(bands_seen))},
                         "frames": int(frames), "partial": len(set(bands_seen)) < 2},
            "canopy": {k: _r(canopy.get(k)) for k in ("height_m", "width_m", "gap_pct", "lean_deg")},
            "symptoms": list(symptoms), "flags": sorted(set(flags)), "detector": list(detector),
            "assisted": list(assisted)}


# ---------------------------------------------------------------------- person events -> hourly aggregate
class PersonHoldCounter:
    """Person events stay on the device (OS 5.6, OQ-31 A); only ``person_hold_count`` per sim hour leaves, with no
    coordinates, no ids and no times finer than the hour."""

    def __init__(self):
        self.by_hour: Dict[int, int] = {}

    def add(self, t_s: float):
        h = int(max(0.0, float(t_s)) // 3600)
        self.by_hour[h] = self.by_hour.get(h, 0) + 1

    def aggregates(self) -> List[dict]:
        return [{"id": "person_hold_count", "hour": h, "count": n} for h, n in sorted(self.by_hour.items())]


def person_cause(alert: dict) -> bool:
    """An alert caused by a person (its reason / class / detail / cause names one): it must not leave per event."""
    for k in ("reason", "class", "cls", "detail", "cause", "kind", "latch_reason", "obstacle", "src"):
        v = alert.get(k)
        if isinstance(v, str) and any(t in PERSONAL_TOKENS for t in key_tokens(v)):
            return True
        if isinstance(v, dict) and person_cause(v):
            return True
    return False


# ---------------------------------------------------------------------- kis/v1 uplink
class Uplink(km.KisMapper):
    """kis/v1 PROPOSED streams of WP9 on top of the WP-telemetry ``KisMapper`` (envelope, sim-* tenant, 16 KiB)."""

    STREAMS = {"tree": ("PROPOSED", "WP9 OS 5.4"), "mark": ("PROPOSED", "WP9 OS 5.4"),
               "mission": ("PROPOSED", "robot-stream-design / WP9"), "alert": ("PROPOSED", "WP9 OS 5.4"),
               "harvest": ("PROPOSED", "atharv-nandi only"), "kpi": ("PROPOSED", "MVP-SPEC 5.7")}

    def __init__(self, tenant: str = TENANT, site: str = "sim-orchard", device: str = "nandi-sim", boot: int = 1):
        super().__init__(tenant, site, device, boot)
        self.persons = PersonHoldCounter()
        self.refused: List[Tuple[str, List[str]]] = []

    def _checked(self, stream: str, ts_ms: int, body: dict, person_event: bool = False):
        bad = contract_violations(body, person_event=person_event)
        if bad:
            self.refused.append((stream, bad))
            raise ValueError(f"tele/{stream} refused: {bad[:4]}")
        return self._tele(stream, ts_ms, body)

    def tree(self, ts_ms: int, rec: dict):
        """tele/tree: the TreeRecord without media paths (manifest root + counts only) and without the leaf
        ``findings[]`` list (count only): no images, leads stay leads."""
        body = copy.deepcopy(rec)
        media = body.pop("media", None) or {}
        body["media"] = {"manifest_root": media.get("manifest_root"),
                         "keyframes_n": media.get("keyframes_n", len(media.get("keyframe_paths") or []))}
        leaf = body.get("leaf")
        if isinstance(leaf, dict) and "findings" in leaf:
            leaf["leads_n"] = len(leaf.pop("findings") or [])
            leaf.pop("revisit_leaves", None)
        body.pop("geo", None)                     # the field frame stays local; the platform maps tree ids
        return self._checked("tree", ts_ms, body)

    def mark(self, ts_ms: int, m: dict):
        body = {k: v for k, v in m.items()}
        errs = validate_mark(m)
        if errs:
            self.refused.append(("mark", errs))
            raise ValueError(f"tele/mark refused: {errs[:4]}")
        return self._tele("mark", ts_ms, body)

    def harvest_v2(self, ts_ms: int, h: dict):
        return self._checked("harvest", ts_ms, dict(h))

    def mission_summary(self, ts_ms: int, summary: dict):
        return self._checked("mission", ts_ms, dict(summary))

    def kpi(self, ts_ms: int, card_rows: dict):
        return self._checked("kpi", ts_ms, dict(card_rows))

    def alert(self, ts_ms: int, alert: dict) -> Optional[Tuple[str, dict]]:
        """One robot alert -> tele/alert, or None. Person events are counted, never sent one by one."""
        aid = str(alert.get("id") or alert.get("event") or alert.get("type") or "")
        if aid in PERSON_ALERTS or "person" in aid or person_cause(alert):
            # a person event under any id (e.g. protective_stop with reason person): hourly count only, OS 5.6
            self.persons.add(float(alert.get("t") or 0.0))
            return None
        kid = ALERT_MAP.get(aid)
        if kid is None:
            return None
        body = {"id": kid, "src": aid, "t_sim": _r(alert.get("t"), 2)}
        for k in ("reason", "attempt", "armed", "assist", "level"):
            if isinstance(alert.get(k), (str, int, float, bool)):
                body[k] = alert[k]
        return self._checked("alert", ts_ms, body)

    def person_aggregates(self, ts_ms: int) -> List[Tuple[str, dict]]:
        return [self._checked("alert", ts_ms, a, person_event=True) for a in self.persons.aggregates()]
