"""WP9 Hindi-first job sheets ``JobSheet nandi.jobsheet/1`` (pure Python, no ROS, no MuJoCo).

Spec: MISSION-PRESETS 0.6 (one sheet per row, Hindi first, English second: ``physical_tag``, side/clock,
``neeche/beech/upar``, job, tool, notes; printable HTML and a Hindi voice list, text only in the sim), 9 step 3
(DOCK/UNLOAD generates the sheets of the rows visited), ORCHARD-MVP-SPEC 1.3 O-23 (every MARK appears), 5.6.

The sheet is built from the MarkRecords (``nandi.mark/1``) and the tree registry entries; it carries no images, no
personal fields and no coordinates beyond the tree's painted tag and the clock / side of the job. Hindi wording is
a first draft for the farm team (assumption: reviewed with the pilot farm crew before the pilot).
"""
from __future__ import annotations

import html
from typing import Dict, Iterable, List, Optional, Sequence

from atharv_nandi.core import records as rec

SCHEMA = "nandi.jobsheet/1"
LANGS = ("hi", "en")

JOB = {
    "fruit": ("फल तोड़ें", "pick fruit"),
    "prune": ("छँटाई करें", "prune"),
    "rootstock_sprout": ("मूलवृंत की कोपल हटाएँ", "remove rootstock sprout"),
    "skirt_lift": ("निचली डाल छाँटें", "lift skirt branch"),
    "malformed_panicle": ("विकृत मंजरी हटाएँ", "remove malformed panicle"),
    "sanitation_deleaf": ("रोगी पत्ते डंठल से काटें", "sanitation de-leaf at the petiole"),
    "sucker": ("पुत्तियाँ (सकर) छाँटें", "de-sucker"),
    "bunch": ("घार काटें", "cut the bunch"),
}
TOOL = {
    "pole_picker": ("बाँस वाला तोड़क", "pole picker"),
    "lopper": ("लॉपर (बड़ी कैंची)", "lopper"),
    "sickle": ("हँसिया", "sickle"),
    "saw": ("आरी", "saw"),
    "human_cut": ("हाथ से कटाई", "hand cut"),
    None: ("हाथ से", "by hand"),
}
HEIGHT = {"neeche": ("नीचे", "low"), "beech": ("बीच", "middle"), "upar": ("ऊपर", "high")}
SIDE = {"N": ("उत्तर", "N"), "E": ("पूर्व", "E"), "S": ("दक्षिण", "S"), "W": ("पश्चिम", "W")}
NOTE = {
    "out_of_reach_marked": ("रोबोट की पहुँच से बाहर", "out of robot reach"),
    "no_clearance": ("जबड़े के लिए जगह कम", "no jaw clearance"),
    "twist_failed": ("मरोड़ने से नहीं टूटा", "twist did not release"),
    "over_capacity": ("डाल मोटी है", "branch too thick for the robot"),
    "paste_required": ("कटे पर बोर्डो पेस्ट लगाएँ", "apply Bordeaux paste on the cut"),
    "bagged": ("थैली लगी है", "bagged"),
    "human_cut": ("मानव कटाई", "human cut"),
    "leaf_minimum": ("कम से कम हरे पत्ते छोड़ें", "keep the minimum green leaves"),
    "bunch_exclusion": ("कटाई के समय 3 मीटर दूर रहें", "keep 3 m away while cutting"),
    "not_in_mvp": ("यह काम अभी रोबोट नहीं करता", "not a robot job yet"),
    "keep_wood": ("यह लकड़ी रखें, न काटें", "keep this wood, do not cut"),
    "operator": ("ऑपरेटर ने चिह्नित किया", "marked by the operator"),
    "sanitation": ("रोग से सफ़ाई", "sanitation"),
    "desucker": ("पुत्ती (सकर) हटाना", "de-suckering"),
    "bunch": ("घार", "bunch"),
}
# fruit stages of the crop profiles (config/crops/*.yaml) and banana sucker types (world/trees.py)
STAGE = {"immature": ("कच्चा", "immature"), "colour_break": ("रंग बदलना शुरू", "colour break"),
         "mature": ("पका (तोड़ने लायक)", "mature"), "ripe": ("पका", "ripe")}
SUCKER = {"peeper": ("छोटी पुत्ती (पीपर)", "peeper"), "sword": ("तलवार पुत्ती", "sword sucker"),
          "water": ("जल पुत्ती (कमज़ोर)", "water sucker")}


def _pair(table: dict, key, default=None):
    v = table.get(key)
    if v is None:
        return (str(key if default is None else default),) * 2
    return v


def _tree_index(entries: Iterable[dict]) -> Dict[str, dict]:
    return {e["tree_id"]: e for e in entries}


def _notes(m: dict) -> List[tuple]:
    out = []
    if m.get("reason") in NOTE:
        out.append(NOTE[m["reason"]])
    if m.get("paste_required") and m.get("reason") != "paste_required":
        out.append(NOTE["paste_required"])
    if m.get("kind") == "bunch":
        out.append(NOTE["bunch_exclusion"])
        if m.get("hands_n_lead") is not None:
            out.append((f"{m['hands_n_lead']} हत्थे (अनुमान)", f"{m['hands_n_lead']} hands (lead)"))
        if m.get("shooting_date"):
            out.append((f"फूल निकलने की तारीख {m['shooting_date']}", f"shooting date {m['shooting_date']}"))
        if m.get("harvest_window"):
            hw = m["harvest_window"]
            frm = hw.get("from") if isinstance(hw, dict) else hw
            out.append((f"कटाई की तारीख {frm}", f"harvest from {frm}"))
        if m.get("ribbon_colour"):
            out.append((f"फीता: {m['ribbon_colour']}", f"ribbon: {m['ribbon_colour']}"))
    if m.get("kind") == "sanitation_deleaf":
        # WP8 banana_marks puts the leaf id in target_id (one MARK per leaf); leaf_ids is the multi-leaf form
        ids = [str(x) for x in (m.get("leaf_ids") or ([m["target_id"]] if m.get("target_id") else []))]
        if ids:
            out.append((f"पत्ता: {', '.join(ids)}", f"leaf: {', '.join(ids)}"))
        if m.get("sigatoka_pct") is not None:
            out.append((f"सिगाटोका {m['sigatoka_pct']}%", f"Sigatoka {m['sigatoka_pct']} %"))
    if m.get("kind") == "sucker" and m.get("sucker_type"):
        out.append(_pair(SUCKER, m["sucker_type"]))
    if m.get("diameter_mm") is not None:
        out.append((f"मोटाई {m['diameter_mm']} मिमी", f"{m['diameter_mm']} mm"))
    if m.get("stage"):
        st_hi, st_en = _pair(STAGE, m["stage"])
        out.append((f"अवस्था: {st_hi}", f"stage: {st_en}"))
    return out


def line_of(m: dict, entry: Optional[dict]) -> dict:
    job_hi, job_en = _pair(JOB, m.get("kind"))
    tool_hi, tool_en = _pair(TOOL, m.get("tool_hint"))
    h_hi, h_en = _pair(HEIGHT, m.get("h_class"))
    s_hi, s_en = _pair(SIDE, m.get("side"))
    notes = _notes(m)
    return {"physical_tag": (entry or {}).get("physical_tag") or m.get("physical_tag"), "tree_id": m.get("tree_id"),
            "side": m.get("side"), "side_hi": s_hi, "clock": m.get("clock"), "h_class": m.get("h_class"),
            "h_class_hi": h_hi, "h_class_en": h_en, "h_m": m.get("h_m"), "job": m.get("kind"), "job_hi": job_hi,
            "job_en": job_en, "tool": m.get("tool_hint"), "tool_hi": tool_hi, "tool_en": tool_en,
            "notes_hi": "; ".join(n[0] for n in notes), "notes_en": "; ".join(n[1] for n in notes),
            "mark_ref": rec.mark_key(m)}


def build_job_sheets(marks: Sequence[dict], entries: Iterable[dict], ident: Dict[str, str],
                     rows: Optional[Sequence[str]] = None, t: Optional[float] = None) -> Dict[str, dict]:
    """One ``nandi.jobsheet/1`` per row: every MARK of the row's trees, in walking order (tree x, then clock).
    ``rows`` = the rows visited (MP 9 step 3); a row with no MARK gets an empty sheet ("no jobs") so the crew sees
    the row was checked. MARKs of a tree outside ``rows`` still get their row's sheet (never dropped, O-23)."""
    idx = _tree_index(entries)
    by_row: Dict[str, List[dict]] = {r: [] for r in (rows or [])}
    for m in rec.dedupe_marks(marks):
        e = idx.get(m.get("tree_id"))
        row = (e or {}).get("row_id") or "unknown"
        by_row.setdefault(row, []).append(m)
    sheets = {}
    for row, ms in by_row.items():
        def order(m):
            e = idx.get(m.get("tree_id")) or {}
            g = e.get("geo") or {}
            return (float(g.get("x", 0.0)), str(m.get("tree_id")), int(m.get("clock") or 0), float(m.get("h_m") or 0))
        lines = [line_of(m, idx.get(m.get("tree_id"))) for m in sorted(ms, key=order)]
        for i, ln in enumerate(lines, 1):
            ln["line_no"] = i
        block = next((idx[m["tree_id"]].get("block_id") for m in ms if m.get("tree_id") in idx), None)
        if block is None:
            block = next((e.get("block_id") for e in idx.values() if e.get("row_id") == row), None)
        sheets[row] = {"schema": SCHEMA, **ident, "row_id": row, "block_id": block, "lang": list(LANGS),
                       "title_hi": f"कतार {row}: काम की सूची", "title_en": f"Row {row}: job sheet",
                       "t_sim": None if t is None else round(float(t), 1), "n_marks": len(lines), "lines": lines,
                       "mark_refs": [ln["mark_ref"] for ln in lines],
                       "columns_hi": ["क्रम", "पेड़ नंबर", "दिशा / घड़ी", "ऊँचाई", "काम", "औज़ार", "टिप्पणी"],
                       "columns_en": ["#", "tree tag", "side / clock", "height", "job", "tool", "notes"]}
    return sheets


def missing_marks(marks: Sequence[dict], sheets: Dict[str, dict]) -> List[str]:
    """O-23: MARK keys that appear on no job sheet ([] = every MARK appears)."""
    have = {k for s in sheets.values() for k in s.get("mark_refs") or []}
    return [rec.mark_key(m) for m in rec.dedupe_marks(marks) if rec.mark_key(m) not in have]


def render_html(sheet: dict) -> str:
    """Printable HTML (A4 portrait, Hindi first, English second). Text only."""
    e = html.escape
    head = "".join(f"<th>{e(h)}<br><small>{e(en)}</small></th>"
                   for h, en in zip(sheet["columns_hi"], sheet["columns_en"]))
    rows = []
    for ln in sheet["lines"]:
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in (
            ln["line_no"], e(str(ln["physical_tag"])),
            f"{e(ln['side_hi'])} / {e(str(ln['clock']))} बजे<br><small>{e(str(ln['side']))} / {e(str(ln['clock']))} o'clock</small>",
            f"{e(ln['h_class_hi'])}<br><small>{e(ln['h_class_en'])}</small>",
            f"{e(ln['job_hi'])}<br><small>{e(ln['job_en'])}</small>",
            f"{e(ln['tool_hi'])}<br><small>{e(ln['tool_en'])}</small>",
            f"{e(ln['notes_hi'])}<br><small>{e(ln['notes_en'])}</small>")) + "</tr>")
    if not rows:
        rows.append('<tr><td colspan="7">इस कतार में कोई काम नहीं<br><small>no jobs in this row</small></td></tr>')
    return ("<!DOCTYPE html><html lang=\"hi\"><head><meta charset=\"utf-8\">"
            f"<title>{e(sheet['title_hi'])} / {e(sheet['title_en'])}</title><style>body{{font-family:'Noto Sans Devanagari',sans-serif;"
            "margin:12mm}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #444;padding:4px;"
            "vertical-align:top}}small{{color:#555}}@page{{size:A4 portrait}}</style></head><body>"
            f"<h1>{e(sheet['title_hi'])}</h1><p><small>{e(sheet['title_en'])} · run {e(sheet['run_id'])} · "
            f"{e(sheet['data_provenance'])} (सिमुलेशन) · {sheet['n_marks']} काम / jobs</small></p>"
            f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>")


def voice_lines(sheet: dict) -> List[str]:
    """Hindi voice list (text for a text-to-speech engine; the sim writes text only)."""
    out = [f"{sheet['title_hi']}। कुल {sheet['n_marks']} काम।"]
    for ln in sheet["lines"]:
        out.append(f"{ln['line_no']}. पेड़ {ln['physical_tag']}, {ln['side_hi']} तरफ़, {ln['clock']} बजे, "
                   f"{ln['h_class_hi']}: {ln['job_hi']}, {ln['tool_hi']}।"
                   + (f" {ln['notes_hi']}।" if ln["notes_hi"] else ""))
    return out


def validate_sheet(sheet: dict) -> List[str]:
    errs = []
    if sheet.get("schema") != SCHEMA:
        errs.append("schema")
    if list(sheet.get("lang") or []) != list(LANGS):
        errs.append("lang_order")
    for ln in sheet.get("lines") or []:
        for k in ("physical_tag", "job_hi", "tool_hi", "h_class_hi", "clock", "side"):
            if ln.get(k) in (None, ""):
                errs.append(f"line{ln.get('line_no')}:missing:{k}")
    errs += rec.contract_violations({k: v for k, v in sheet.items() if k != "lines"})
    return errs
