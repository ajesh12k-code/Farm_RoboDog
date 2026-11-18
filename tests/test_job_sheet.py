"""WP9 Hindi-first job sheets (MP 0.6, MP 9 step 3, O-23: every MARK appears). ROS-free, MuJoCo-free."""
import json
import os

from atharv_nandi.core import job_sheet as js
from atharv_nandi.core import records as rec
from atharv_nandi.core.tasks import common as tc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IDENT = rec.identity("run-js-1", "nandi-sim", "nandi-sim-20261118T0900Z")


def field():
    return json.load(open(os.path.join(REPO, "config", "fields", "orchard_demo.json")))


def tree(entries, tid):
    e = next(t for t in entries if t["tree_id"] == tid)
    return {"tree_id": tid, "physical_tag": e["physical_tag"], "pos": (e["geo"]["x"], e["geo"]["y"])}


def demo_marks(entries):
    """The OS 1.2 MARK set: out-of-band mango, guava 20 mm skirt branch + water shoot, papaya no_clearance, banana
    deleaf + bunch + 2 suckers."""
    ot, g2, p1, b2 = (tree(entries, t) for t in ("OT-1", "R2-2", "R3-1", "R4-2"))
    return [
        tc.make_mark(ot, "fruit", (6.6, -3.0, 1.6), "out_of_reach_marked", stage="mature", tool_hint="pole_picker",
                     target_id="OT-1:f11"),
        tc.make_mark(ot, "fruit", (5.4, -3.9, 2.1), "out_of_reach_marked", stage="mature", tool_hint="pole_picker",
                     target_id="OT-1:f12"),
        tc.make_mark(g2, "skirt_lift", (3.7, 7.0, 0.5), "over_capacity", diameter_mm=20.0, tool_hint="lopper",
                     target_id="R2-2:prune:3"),
        tc.make_mark(g2, "prune", (3.5, 7.3, 1.6), "out_of_reach_marked", diameter_mm=9.0, tool_hint="pole_picker",
                     target_id="R2-2:prune:5"),
        tc.make_mark(p1, "fruit", (2.1, 10.8, 0.7), "no_clearance", stage="colour_break", tool_hint="human_cut",
                     target_id="R3-1:f02"),
        tc.make_mark(b2, "sanitation_deleaf", (3.9, 14.2, 1.4), "human_cut", tool_hint="sickle",
                     target_id="R4-2:leaf:3", extra={"leaf_ids": [3, 5]}),
        tc.make_mark(b2, "bunch", (3.8, 14.3, 1.5), "human_cut", tool_hint="human_cut", target_id="R4-2:bunch",
                     extra={"ribbon_colour": "blue", "harvest_window": {"from": "2026-12-20", "to": "2027-01-15"}}),
        tc.make_mark(b2, "sucker", (3.6, 14.6, 0.1), "human_cut", tool_hint="sickle", target_id="R4-2:sucker:1"),
        tc.make_mark(b2, "sucker", (4.0, 14.7, 0.1), "human_cut", tool_hint="sickle", target_id="R4-2:sucker:2"),
    ]


def test_one_sheet_per_row_every_mark_appears_hindi_first():
    f = field()
    marks = demo_marks(f["trees"])
    sheets = js.build_job_sheets(marks + [dict(marks[0])], f["trees"], IDENT, rows=["OT", "R1", "R2", "R3", "R4"])
    assert set(sheets) == {"OT", "R1", "R2", "R3", "R4"}
    assert sheets["R1"]["n_marks"] == 0                          # visited, no jobs: still a sheet
    assert js.missing_marks(marks, sheets) == []                  # O-23
    assert sum(s["n_marks"] for s in sheets.values()) == len(marks)   # the duplicate MARK appears once
    for s in sheets.values():
        assert s["lang"] == ["hi", "en"] and s["schema"] == "nandi.jobsheet/1"
        assert js.validate_sheet(s) == []
    r4 = sheets["R4"]["lines"]
    assert [ln["job"] for ln in r4] == ["bunch", "sanitation_deleaf", "sucker", "sucker"] or len(r4) == 4
    bunch = next(ln for ln in r4 if ln["job"] == "bunch")
    assert "3 मीटर" in bunch["notes_hi"] and "2026-12-20" in bunch["notes_en"]
    skirt = next(ln for ln in sheets["R2"]["lines"] if ln["job"] == "skirt_lift")
    assert "बोर्डो पेस्ट" in skirt["notes_hi"] and skirt["tool_hi"].startswith("लॉपर")


def test_mark_of_an_unvisited_row_is_never_dropped():
    f = field()
    marks = demo_marks(f["trees"])
    sheets = js.build_job_sheets(marks, f["trees"], IDENT, rows=["OT"])
    assert js.missing_marks(marks, sheets) == [] and "R4" in sheets


def test_missing_mark_is_reported():
    f = field()
    marks = demo_marks(f["trees"])
    sheets = js.build_job_sheets(marks[:-1], f["trees"], IDENT)
    assert js.missing_marks(marks, sheets) == [rec.mark_key(marks[-1])]


def test_html_and_voice_are_text_only_and_hindi_first():
    f = field()
    sheets = js.build_job_sheets(demo_marks(f["trees"]), f["trees"], IDENT)
    page = js.render_html(sheets["R2"])
    assert page.startswith("<!DOCTYPE html>") and 'lang="hi"' in page
    assert page.index("कतार R2") < page.index("Row R2")
    assert "<img" not in page and "base64" not in page and "<script" not in page
    voice = js.voice_lines(sheets["R4"])
    assert voice[0].startswith("कतार R4") and len(voice) == 1 + sheets["R4"]["n_marks"]
    assert "Ram" not in "".join(voice)
    empty = js.render_html(js.build_job_sheets([], f["trees"], IDENT, rows=["R1"])["R1"])
    assert "कोई काम नहीं" in empty


def test_sheet_carries_no_personal_or_image_field():
    f = field()
    for s in js.build_job_sheets(demo_marks(f["trees"]), f["trees"], IDENT).values():
        assert rec.contract_violations(s) == []
