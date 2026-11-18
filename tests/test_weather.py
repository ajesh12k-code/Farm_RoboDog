"""WP7 weather stub: diurnal UP temperature, rain and wet-foliage timing, per-preset weather verdicts
(ORCHARD-MVP-SPEC 4.5), and the weather path through the platform hook (ROS-free, each <= 10 s)."""
import os
import sys

import pytest


from atharv_nandi.core import weather_stub as ws  # noqa: E402
from test_platform import Run, mb  # noqa: E402


def test_diurnal_temperature_follows_the_monthly_normals():
    for month, (tmax, tmin) in ws.UP_MONTHLY_MAX_MIN_C.items():
        assert ws.diurnal_temp_c(month, 6.0) == pytest.approx(tmin)
        assert ws.diurnal_temp_c(month, 15.0) == pytest.approx(tmax)
        hours = [h / 4 for h in range(96)]
        temps = [ws.diurnal_temp_c(month, h) for h in hours]
        assert min(temps) >= tmin - 1e-9 and max(temps) <= tmax + 1e-9
        steps = [abs(b - a) for a, b in zip(temps, temps[1:] + temps[:1])]
        assert max(steps) < (tmax - tmin) * 0.2          # continuous, incl. across midnight
    assert ws.diurnal_temp_c(5, 14.0) > ws.HEAT_DEGRADED_C           # May afternoon: heat degraded
    assert ws.diurnal_temp_c(1, 9.0) < 20.0


def test_rain_and_wet_foliage_timing():
    st = ws.WeatherStub(ws.WeatherScript(month=8, start_hour=7.0, rain=[ws.RainEvent(600.0, 1200.0, 8.0)]))
    assert st.state(0.0)["rain"] is False and st.state(0.0)["wet_foliage"] is False
    s = st.state(900.0)
    assert s["rain"] and s["rain_mm_h"] == 8.0 and s["wet_foliage"] and s["wet_cause"] == "rain"
    s = st.state(1800.0 + 2 * 3600.0 - 1.0)              # dries 2 h after the rain stopped
    assert not s["rain"] and s["wet_foliage"]
    assert st.state(1800.0 + 2 * 3600.0 + 1.0)["wet_foliage"] is False
    assert st.state(0.0)["dawn_mode_window"] and not st.state(3600.0 * 1.5)["dawn_mode_window"]
    dew = ws.WeatherStub(ws.WeatherScript(month=1, start_hour=6.0, dew=True))
    assert dew.state(0.0)["wet_cause"] == "dew" and not dew.state(3 * 3600.0)["wet_foliage"]
    wet0 = ws.WeatherStub(ws.WeatherScript(wet_at_start=True))
    assert wet0.state(10.0)["wet_foliage"] and not wet0.state(3 * 3600.0)["wet_foliage"]
    st.set_rain(5000.0, 60.0)
    assert st.state(5030.0)["rain"]
    st.reset()
    assert not st.state(5030.0)["rain"]


def test_preset_verdicts():
    def state(**kw):
        base = {"rain": False, "wet_foliage": False, "temp_c": 30.0}
        return {**base, **kw}
    assert ws.preset_verdict(state(), "scan") == {"level": "go", "reasons": []}
    v = ws.preset_verdict(state(rain=True, wet_foliage=True), "scan")
    assert v["level"] == "no_go" and [r["code"] for r in v["reasons"]] == ["rain"]
    # wet foliage without rain: SCAN / TREE 360 go, LEAF SCAN and contact presets no-go
    assert ws.preset_verdict(state(wet_foliage=True), "tree360")["level"] == "go"
    for p in ("leaf_scan", "harvest", "prune", "work"):
        v = ws.preset_verdict(state(wet_foliage=True), p)
        assert v["level"] == "no_go" and v["reasons"] == [{"code": "wet_foliage", "level": "no_go"}]
    heat = {"level": "degraded", "reasons": [{"code": "heat", "level": "degraded"}]}
    assert ws.preset_verdict(state(temp_c=39.0), "scan") == heat
    assert ws.preset_verdict(state(temp_c=38.0), "scan")["level"] == "go"          # > 38, not >=
    assert ws.preset_verdict(state(temp_c=42.5), "leaf_scan")["level"] == "no_go"
    # recovery, dock and return home are never blocked by weather
    for p in ("return_home", "safe_park", "dock", "recovery"):
        assert ws.preset_verdict(state(rain=True, wet_foliage=True, temp_c=45.0), p)["level"] == "go"


def test_script_parse_and_validation():
    s = ws.WeatherScript.from_dict({"month": "5", "start_hour": 13, "temp_offset_c": 3, "dew": 1,
                                    "rain": [{"t0": 10, "dur": 20}], "temp_c": None, "unknown": 1})
    assert (s.month, s.start_hour, s.temp_offset_c, s.dew, s.temp_c) == (5, 13.0, 3.0, True, None)
    assert s.rain[0].mm_h == 5.0
    st = ws.WeatherStub(s).state(0.0)
    assert st["temp_c"] == pytest.approx(ws.diurnal_temp_c(5, 13.0) + 3.0, abs=0.01)
    assert ws.WeatherStub(ws.WeatherScript(temp_c=44.0)).state(0.0)["temp_c"] == 44.0
    assert ws.WeatherScript(start_hour=23.5).hour(3600.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        ws.WeatherScript(month=13)
    with pytest.raises(ValueError):
        ws.WeatherScript(start_hour=24.0)
    with pytest.raises(ValueError):
        ws.RainEvent(0.0, 0.0)


def test_weather_through_the_hook(monkeypatch):
    monkeypatch.setenv("NANDI_WEATHER", '{"month": 6, "start_hour": 14.0}')
    m = mb.build(NANDI_TERRAIN="flat", NANDI_CROPS=1)
    r = Run(m)
    r.run(1.0)
    w = r.bus.values["weather"]
    assert w["month"] == 6 and w["temp_c"] == pytest.approx(ws.diurnal_temp_c(6, 14.0), abs=0.05)
    assert ws.preset_verdict(w, "scan")["level"] == "degraded"            # June afternoon 38.7 C
    r.bus.deliver("/nandi/weather/cmd", {"rain": {"dur": 30.0, "mm_h": 12.0}, "temp_c": 25.0})
    r.run(1.0)
    w = r.bus.values["weather"]
    assert w["rain"] and w["rain_mm_h"] == 12.0 and w["temp_c"] == 25.0
    assert ws.preset_verdict(w, "leaf_scan")["level"] == "no_go"
    pubs = r.msgs("/nandi/weather")
    assert len(pubs) in (2, 3) and pubs[-1]["rain"] is True and pubs[-1]["source"] == "stub"


def test_preset_classes_cover_per_crop_ids():
    wet = {"rain": False, "wet_foliage": True, "temp_c": 30.0}
    for p in ("harvest_mango", "harvest_guava", "clip_prune", "leaf_scan_dawn", "work_assist", "pick_papaya",
              "brand_new_preset"):
        assert ws.preset_verdict(wet, p)["level"] == "no_go", p
    for p in ("scan", "scan_row", "tree360", "tree_360_video", "trunk_base_scan", "fruit_scout", "mark_banana"):
        assert ws.preset_verdict(wet, p)["level"] == "go", p
    for p in ("dock_unload", "recovery_self_right", "return_home"):
        assert ws.preset_class(p) == "exempt"
    assert ws.preset_class("brand_new_preset") == "unknown"
    assert ws.preset_class("scanner") == "unknown"              # prefix needs the "_" separator


def test_weather_reset_drops_runtime_temp():
    st = ws.WeatherStub(ws.WeatherScript(temp_c=None))
    st.set_temp(45.0)
    assert st.state(0.0)["temp_c"] == 45.0
    st.reset()
    assert st.script.temp_c is None
    with pytest.raises(ValueError):
        st.set_temp(float("nan"))
