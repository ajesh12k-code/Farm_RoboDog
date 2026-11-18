"""WP10 console layout and language checks (static, ROS-free): E-STOP on every view and >= 64 px on phones,
360 px phones without horizontal overflow, Hindi first, en / hi complete, every element the scripts use exists."""
import os
import re
from html.parser import HTMLParser

import pytest

from atharv_nandi.core import console_protocol as cp
from atharv_nandi.core import ready_check as rc
from atharv_nandi.core import records as rec

WEB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "atharv_nandi", "web"))
PHONE_W = 360
VIEWS = ("map", "tree", "queue", "recovery", "jobs", "drive")


def _read(name):
    with open(os.path.join(WEB, name), encoding="utf-8") as fh:
        return fh.read()


class _Tree(HTMLParser):
    """Element list with each element's ancestor chain (ids, data-view)."""

    VOID = {"meta", "link", "input", "img", "br", "hr", "source"}

    def __init__(self):
        super().__init__()
        self.stack, self.elements = [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        el = {"tag": tag, "attrs": a, "ancestors": [dict(x["attrs"], tag=x["tag"]) for x in self.stack]}
        self.elements.append(el)
        if tag not in self.VOID:
            self.stack.append(el)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i]["tag"] == tag:
                del self.stack[i:]
                break


@pytest.fixture(scope="module")
def page():
    p = _Tree()
    p.feed(_read("index.html"))
    return p


@pytest.fixture(scope="module")
def css():
    return _read("style.css")


def _blocks(css_text):
    """(media condition or '', selector, declarations) for every rule."""
    out, i, media = [], 0, ""
    text = re.sub(r"/\*.*?\*/", "", css_text, flags=re.S)
    depth_media = []
    for m in re.finditer(r"(@media[^{]+)\{|([^{}@][^{}]*)\{([^{}]*)\}|\}", text):
        if m.group(1):
            depth_media.append(m.group(1).strip())
        elif m.group(2) is not None:
            out.append((depth_media[-1] if depth_media else "", m.group(2).strip(), m.group(3)))
        elif depth_media:
            depth_media.pop()
    return out


def _px(decl, prop):
    m = re.search(rf"(?:^|;)\s*{prop}\s*:\s*([0-9.]+)px", decl)
    return float(m.group(1)) if m else None


def _var(css_text, name, media_sub=None):
    vals = []
    for media, sel, decl in _blocks(css_text):
        if sel == ":root" and (media_sub is None or media_sub in media) and (media_sub is not None or not media):
            m = re.search(rf"--{name}\s*:\s*([0-9.]+)px", decl)
            if m:
                vals.append(float(m.group(1)))
    return vals


# ---------------------------------------------------------------- E-STOP

def test_estop_is_outside_every_view_and_never_inside_the_sheet(page):
    btn = [e for e in page.elements if e["attrs"].get("id") == "btn-estop"]
    assert len(btn) == 1
    chain = btn[0]["ancestors"]
    assert [a.get("tag") for a in chain] == ["html", "body", "div"] and chain[-1].get("id") == "estop-box"
    assert not any("data-view" in a or a.get("id") in ("sheet", "tabs") for a in chain)
    views = {e["attrs"]["data-view"] for e in page.elements if "data-view" in e["attrs"]}
    assert views == set(VIEWS)
    tabs = {e["attrs"]["data-tab"] for e in page.elements if e["tag"] == "button" and "data-tab" in e["attrs"]}
    assert tabs == set(VIEWS) | {"3d"}


def test_estop_is_at_least_64px_everywhere_and_on_top(css):
    base = _var(css, "estop")
    phone = _var(css, "estop", "max-width: 760px")
    assert base and phone and min(base + phone) >= 64, (base, phone)
    rules = [(m, d) for m, s, d in _blocks(css) if s == "button.estop"]
    assert rules and _px(rules[0][1], "min-width") >= 64 and _px(rules[0][1], "min-height") >= 64
    assert "var(--estop)" in rules[0][1]
    z = {s: int(re.search(r"z-index\s*:\s*(\d+)", d).group(1)) for _, s, d in _blocks(css) if "z-index" in d}
    assert z["#estop-box"] == max(z.values())
    box = [d for m, s, d in _blocks(css) if s == "#estop-box" and "760" in m]
    assert box and "right" in box[0] and "bottom" in box[0]           # phones: thumb corner, above the bottom nav
    # nothing hides the E-STOP on any tab
    for m, s, d in _blocks(css):
        if any(k in s for k in ("#estop-box", "button.estop", "#btn-estop")) and re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", d):
            pytest.fail(f"{s} hides the E-STOP")


# ---------------------------------------------------------------- 360 px phones

def test_phone_layout_has_no_fixed_width_wider_than_360px(css):
    blocks = _blocks(css)
    phone = {s: d for m, s, d in blocks if "max-width: 760px" in m}
    for media, sel, decl in blocks:
        if media:
            continue
        for prop in ("width", "min-width"):
            v = _px(decl, prop)
            if v is not None and v > PHONE_W:
                # every desktop width above 360 px is overridden on phones
                assert sel in phone and re.search(r"(^|;)\s*(width|left)\s*:", phone[sel]), (sel, prop, v)
    for sel, decl in phone.items():
        for prop in ("width", "min-width"):
            v = _px(decl, prop)
            assert v is None or v <= PHONE_W, (sel, prop, v)
    # sheet and nav span the screen on phones; the sheet never scrolls sideways
    assert "left: 0" in phone["#sheet"] and "width: auto" in phone["#sheet"]
    assert "left: 0" in phone["#tabs"] and "right: 0" in phone["#tabs"]
    sheet = [d for m, s, d in blocks if s == "#sheet" and not m][0]
    assert "overflow-x: hidden" in sheet
    # fixed-column grids fit: d-pad 3 x 54 + gaps, tree-card buttons and sliders shrink (minmax(0, 1fr))
    dpad = [d for m, s, d in blocks if s == ".dpad" and "760" in m][0]
    assert 3 * int(re.search(r"repeat\(3, (\d+)px\)", dpad).group(1)) + 2 * 6 <= PHONE_W - 16
    assert "minmax(0, 1fr)" in [d for m, s, d in blocks if s == ".tc-buttons"][0]


def test_viewport_meta_and_hindi_document(page):
    metas = [e["attrs"] for e in page.elements if e["tag"] == "meta"]
    assert any(m.get("name") == "viewport" and "width=device-width" in m.get("content", "") for m in metas)
    html = [e for e in page.elements if e["tag"] == "html"][0]
    assert html["attrs"].get("lang") == "hi"


# ---------------------------------------------------------------- languages

def _strings():
    src = _read("i18n.js")
    body = src[src.index("export const STRINGS"):src.index("// MISSION-PRESETS 0.5 reason codes")]
    hi = body[body.index("hi: {"):body.index("en: {")]
    en = body[body.index("en: {"):]
    keys = lambda s: set(re.findall(r"(?:^|[\s,{])([a-z][a-z0-9_]*)\s*:\s*['\"`]", s))  # noqa: E731
    return src, keys(hi), keys(en)


def test_hindi_first_and_both_languages_complete(page):
    src, hi, en = _strings()
    assert src.index("hi: {") < src.index("en: {")
    assert "let lang = 'hi'" in src                               # Hindi is the default language
    assert hi == en, (sorted(hi - en), sorted(en - hi))
    used = {e["attrs"]["data-i18n"] for e in page.elements if "data-i18n" in e["attrs"]}
    js = "".join(_read(f) for f in os.listdir(WEB) if f.endswith(".js"))
    used |= set(re.findall(r"\bt\('([a-z0-9_]+)'\)", js))
    used |= {f"kind_{k}" for k in rec.MARK_KINDS} | {f"h_{h}" for h in rec.H_CLASSES}
    used |= {f"chk_{c}" for c in cp.FALLEN_CHECKS} | {f"band_{b}" for b in ("low", "mid", "high")}
    used |= {c["id"] for c in ({"id": x} for x in ("rtk", "soc", "wind", "slope", "sim_handheld"))}
    missing = sorted(k for k in used if k not in hi)
    assert not missing, missing


def test_every_reason_code_has_hindi_and_english():
    src = _read("i18n.js")
    reasons = src[src.index("export const REASONS"):src.index("const STORE_KEY")]
    have = set(re.findall(r"(?:^|\s)([a-z_]+): \['", reasons))
    from atharv_nandi.core.workflow import reasons as wr
    mp05 = set(wr.MP_05)                       # the MISSION-PRESETS 0.5 reason-code enum
    assert len(mp05) >= 30
    need = mp05 | set(rc.WP27_REASONS) | {"no_live_executor", "estop"}
    assert not sorted(need - have), sorted(need - have)
    for code, pair in re.findall(r"(?:^|\s)([a-z_]+): \[('[^']*', '[^']*')\]", reasons):
        hi_txt, en_txt = re.findall(r"'([^']*)'", pair)
        assert hi_txt and en_txt and re.search(r"[ऀ-ॿ]|RTK|MVP|noVNC", hi_txt), code


# ---------------------------------------------------------------- wiring

def test_every_element_the_scripts_use_exists(page):
    ids = {e["attrs"]["id"] for e in page.elements if "id" in e["attrs"]}
    js = "".join(_read(f) for f in os.listdir(WEB) if f.endswith(".js"))
    used = set(re.findall(r"\$\('([A-Za-z0-9_-]+)'\)", js))
    assert used and not sorted(used - ids), sorted(used - ids)


def test_tree_card_buttons_and_console_panels(page):
    tc = [e["attrs"]["data-tc"] for e in page.elements if "data-tc" in e["attrs"]]
    assert tc == ["tree360", "leaf_scan", "trunk_scan", "fruit_scout", "harvest", "prune", "mark"]   # OS 5.5
    ids = {e["attrs"].get("id") for e in page.elements}
    for need in ("map", "presetbar", "queue-list", "band-strips", "crown-gauge", "progress-chips", "rec-facts",
                 "fallen-checks", "job-row", "btn-job-hi", "assist-dialog", "mark-dialog", "btn-pause", "btn-resume",
                 "btn-abort"):
        assert need in ids, need
    rec_ops = {e["attrs"]["data-rec"] for e in page.elements if "data-rec" in e["attrs"]}
    assert rec_ops <= set(cp.RECOVERY_OPS) and {"run_safe_park", "run_return_home", "abort"} <= rec_ops
    assists = {e["attrs"]["data-assist"] for e in page.elements if "data-assist" in e["attrs"]}
    assert assists == set(cp.ASSIST_REPLIES)
    # operator-only controls are marked so the viewer role greys them out (the bridge refuses them anyway)
    for e in page.elements:
        a = e["attrs"]
        if any(k in a for k in ("data-tc", "data-rec", "data-crown", "data-assist")):
            assert "data-op" in a, a


def test_console_messages_the_browser_sends_are_the_protocol_types():
    js = "".join(_read(f) for f in os.listdir(WEB) if f.endswith(".js"))
    sent = set(re.findall(r"type: '([a-z_]+)'", js))
    known = set(cp.CONSOLE_TYPES) | {"cmd", "service", "gait_params", "manual_arm", "envelope", "ping", "auth",
                                     "mission_start"}
    sent -= {"range"}  # an <input type: range> in controls.js
    assert sent <= known, sorted(sent - known)
    assert set(cp.CONSOLE_TYPES) <= sent | {"preset_resume"}
