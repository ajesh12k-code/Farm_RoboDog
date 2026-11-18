"""Orchard lighting presets and fog (WP4; ORCHARD-MVP-SPEC 3.6, 4.5; MVP-SPEC 4.2 dusk stub).

Orchard worlds name the robot MJCF's light ``sun`` and make it directional (the legacy spot light at 4 m lit a
~4 m disc, too little for a 20 m orchard). A preset sets the sun direction and colour, the headlight and the
fog; switching at runtime writes ``model.light_*`` / ``model.vis.headlight`` / ``model.vis.map.fog*`` and is
carried to renderer processes by the ``lighting`` sim-state mutation provider (``atharv_nandi/sim/hooks/world_events.py``).
Fog reaches the pixels only through ``sync_renderer`` (scene fog flag + MjrContext fog range), which
``camera_rig.CameraRenderer`` and ``sim_driver.MjRenderer`` call before every render.

``level`` is a relative light level (noon 1.0) for the Ready check: camera presets need ``level >=
CAMERA_MIN_LEVEL``; LEAF SCAN dawn mode (LED on, OS 4.5) lowers it to ``DAWN_MODE_MIN_LEVEL``. ``lux`` is an
order-of-magnitude label only. ``camera_noise_sigma`` / ``motion_blur_px`` are the dusk degradations
``perception_node`` applies (MVP-SPEC 4.2: sigma 8/255, 3 px). All values are ASSUMED.
"""
import math
from typing import Dict, Optional

SUN_LIGHT = "sun"
DEFAULT_PRESET = "morning"
CAMERA_MIN_LEVEL = 0.30      # ASSUMED Ready-check threshold for camera presets
DAWN_MODE_MIN_LEVEL = 0.08   # ASSUMED: LEAF SCAN dawn mode with the LED on (OS 4.5)


def _dir(elev_deg: float, az_deg: float):
    """Unit light direction (pointing down from the sun) for a sun at elevation / azimuth (az 0 = +x east)."""
    e, a = math.radians(elev_deg), math.radians(az_deg)
    v = (-math.cos(e) * math.cos(a), -math.cos(e) * math.sin(a), -math.sin(e))
    return [round(c, 6) for c in v]


PRESETS: Dict[str, dict] = {
    "morning": {"elev_deg": 30.0, "az_deg": 0.0, "diffuse": [0.85, 0.80, 0.70], "ambient": [0.25, 0.25, 0.25],
                "specular": [0.2, 0.2, 0.2], "headlight_diffuse": [0.30, 0.30, 0.30],
                "headlight_ambient": [0.15, 0.15, 0.15], "level": 0.7, "lux": 20000},
    "noon": {"elev_deg": 80.0, "az_deg": 180.0, "diffuse": [1.0, 1.0, 0.95], "ambient": [0.30, 0.30, 0.30],
             "specular": [0.3, 0.3, 0.3], "headlight_diffuse": [0.30, 0.30, 0.30],
             "headlight_ambient": [0.15, 0.15, 0.15], "level": 1.0, "lux": 80000},
    "overcast": {"elev_deg": 60.0, "az_deg": 90.0, "diffuse": [0.45, 0.45, 0.47], "ambient": [0.45, 0.45, 0.47],
                 "specular": [0.05, 0.05, 0.05], "headlight_diffuse": [0.25, 0.25, 0.25],
                 "headlight_ambient": [0.20, 0.20, 0.20], "level": 0.5, "lux": 10000},
    "dusk": {"elev_deg": 5.0, "az_deg": 180.0, "diffuse": [0.25, 0.17, 0.11], "ambient": [0.08, 0.07, 0.08],
             "specular": [0.02, 0.02, 0.02], "headlight_diffuse": [0.06, 0.06, 0.06],
             "headlight_ambient": [0.04, 0.04, 0.05], "level": 0.08, "lux": 400,
             "camera_noise_sigma": 8.0 / 255.0, "motion_blur_px": 3},
    "dawn": {"elev_deg": 5.0, "az_deg": 0.0, "diffuse": [0.22, 0.20, 0.26], "ambient": [0.10, 0.10, 0.12],
             "specular": [0.02, 0.02, 0.02], "headlight_diffuse": [0.07, 0.07, 0.08],
             "headlight_ambient": [0.05, 0.05, 0.06], "level": 0.10, "lux": 500,
             "camera_noise_sigma": 6.0 / 255.0, "motion_blur_px": 2, "dawn_mode_window": True},
}
for _p in PRESETS.values():
    _p["dir"] = _dir(_p["elev_deg"], _p["az_deg"])
    _p.setdefault("camera_noise_sigma", 0.0)
    _p.setdefault("motion_blur_px", 0)
    _p.setdefault("dawn_mode_window", False)

FOG_RGBA = [0.78, 0.80, 0.82, 1.0]


def preset(name: str) -> dict:
    if name not in PRESETS:
        raise KeyError(f"lighting preset {name!r} not in {sorted(PRESETS)}")
    return PRESETS[name]


def fog_range(density: float) -> Optional[tuple]:
    """(fogstart, fogend) in units of the model extent for a density in [0, 1]; None = no fog."""
    d = float(density)
    if d <= 0.0:
        return None
    d = min(d, 1.0)
    return (max(0.0, 0.6 * (1.0 - d)), max(0.05, 1.5 * (1.0 - d) + 0.05))


def light_ok(level: float, dawn_mode: bool = False) -> bool:
    """Camera-preset Ready check: enough light (dawn mode lowers the threshold for LEAF SCAN)."""
    return float(level) >= (DAWN_MODE_MIN_LEVEL if dawn_mode else CAMERA_MIN_LEVEL)


def state(name: str, fog: float = 0.0) -> dict:
    """The world-state block for the console / Ready check / perception."""
    p = preset(name)
    return {"preset": name, "level": p["level"], "lux": p["lux"], "fog": float(fog),
            "camera_ok": light_ok(p["level"]), "dawn_mode_ok": light_ok(p["level"], True),
            "dawn_mode_window": p["dawn_mode_window"], "camera_noise_sigma": p["camera_noise_sigma"],
            "motion_blur_px": p["motion_blur_px"], "assumed": True}


# ---------------------------------------------------------------------- MuJoCo side (lazy import)
def sun_id(model) -> int:
    import mujoco
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, SUN_LIGHT)


def apply_spec(spec, name: str):
    """Compose time: name the first light ``sun``, make it directional and set the preset (orchard only)."""
    p = preset(name)
    if not spec.lights:
        spec.worldbody.add_light(name=SUN_LIGHT, pos=[0, 0, 10])
    light = spec.lights[0]
    light.name = SUN_LIGHT
    import mujoco
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.pos = [0.0, 0.0, 10.0]
    light.dir = list(p["dir"])
    light.diffuse = list(p["diffuse"])
    light.ambient = list(p["ambient"])
    light.specular = list(p["specular"])
    light.castshadow = False      # OS 3.7: shadows off
    spec.visual.headlight.diffuse = list(p["headlight_diffuse"])
    spec.visual.headlight.ambient = list(p["headlight_ambient"])


def apply_model(model, name: str, fog: float = 0.0) -> bool:
    """Runtime: write the preset into a compiled model. False when the model has no ``sun`` light."""
    lid = sun_id(model)
    if lid < 0:
        return False
    p = preset(name)
    model.light_dir[lid] = p["dir"]
    model.light_diffuse[lid] = p["diffuse"]
    model.light_ambient[lid] = p["ambient"]
    model.light_specular[lid] = p["specular"]
    model.vis.headlight.diffuse[:] = p["headlight_diffuse"]
    model.vis.headlight.ambient[:] = p["headlight_ambient"]
    fr = fog_range(fog)
    if fr is not None:
        model.vis.map.fogstart, model.vis.map.fogend = fr
        model.vis.rgba.fog[:] = FOG_RGBA
    return True


def snapshot(model) -> Optional[dict]:
    """Every model field a preset writes (for reset and for the diff-equal check)."""
    lid = sun_id(model)
    if lid < 0:
        return None
    return {"dir": model.light_dir[lid].tolist(), "diffuse": model.light_diffuse[lid].tolist(),
            "ambient": model.light_ambient[lid].tolist(), "specular": model.light_specular[lid].tolist(),
            "headlight_diffuse": model.vis.headlight.diffuse.tolist(),
            "headlight_ambient": model.vis.headlight.ambient.tolist(),
            "fogstart": float(model.vis.map.fogstart), "fogend": float(model.vis.map.fogend),
            "fog_rgba": model.vis.rgba.fog.tolist()}


def restore(model, snap: Optional[dict]):
    lid = sun_id(model)
    if lid < 0 or not snap:
        return
    model.light_dir[lid] = snap["dir"]
    model.light_diffuse[lid] = snap["diffuse"]
    model.light_ambient[lid] = snap["ambient"]
    model.light_specular[lid] = snap["specular"]
    model.vis.headlight.diffuse[:] = snap["headlight_diffuse"]
    model.vis.headlight.ambient[:] = snap["headlight_ambient"]
    model.vis.map.fogstart, model.vis.map.fogend = snap["fogstart"], snap["fogend"]
    model.vis.rgba.fog[:] = snap["fog_rgba"]


# ---------------------------------------------------------------------- renderers (fog reaches the pixels)
FOG_ACTIVE_MAX_START = 1.0     # fog_range() never starts fog beyond 0.6 extents; compiled models use 3.0


def fog_active(model) -> bool:
    """Is a fog density > 0 applied to this model? (``fog_range`` puts ``fogstart`` <= 0.6; the MuJoCo default
    and every compiled field world keep 3.0.) Read from the model, so it holds after a sim-state transfer."""
    return sun_id(model) >= 0 and float(model.vis.map.fogstart) < FOG_ACTIVE_MAX_START


def _fog_want(model):
    ext = float(model.stat.extent)
    return (ext * float(model.vis.map.fogstart), ext * float(model.vis.map.fogend),
            [float(v) for v in model.vis.rgba.fog])


def sync_renderer(renderer, model=None) -> bool:
    """Carry the model fog to a ``mujoco.Renderer`` (call before ``update_scene`` / ``render``). Returns
    ``fog_active``.

    MuJoCo renders fog only with ``scene.flags[mjRND_FOG]`` on (off by default), and ``mjr_makeContext`` copies
    the fog range and colour into GL state once: writing ``model.vis.map.fog*`` (preset / fog command / the
    ``lighting`` provider) after the renderer exists changes nothing on screen. So the flag follows
    ``fog_active`` and, when the model fog differs from the context's, the ``MjrContext`` is rebuilt (rare:
    on a fog change). Segmentation passes keep fog off (it would tint the segment ids). Models without a
    ``sun`` light (legacy) are left untouched."""
    import mujoco
    m = model if model is not None else getattr(renderer, "model", None)
    scene = getattr(renderer, "scene", None)
    if m is None or scene is None or sun_id(m) < 0:
        return False
    on = fog_active(m)
    flag = mujoco.mjtRndFlag.mjRND_FOG
    scene.flags[flag] = 1 if (on and not getattr(renderer, "_segmentation_rendering", False)) else 0
    ctx = getattr(renderer, "_mjr_context", None)
    if not on or ctx is None:
        return on
    start, end, rgba = _fog_want(m)
    have = (float(ctx.fogStart), float(ctx.fogEnd), [float(v) for v in ctx.fogRGBA])
    if abs(have[0] - start) <= 1e-4 * max(1.0, start) and abs(have[1] - end) <= 1e-4 * max(1.0, end) \
            and all(abs(a - b) <= 1e-4 for a, b in zip(have[2], rgba)):
        return on
    gl = getattr(renderer, "_gl_context", None)
    if gl is not None:
        gl.make_current()
    new = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    # the Renderer switches its context to reversed-Z depth (mjDEPTH_ZEROFAR) after creating it; a fresh
    # context comes back as mjDEPTH_ZERONEAR, and the one-pass rgbd depth would then be wrong metric depth
    # without any error. Carry the depth mapping (and the offscreen buffer) over to the rebuilt context.
    new.readDepthMap = ctx.readDepthMap
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, new)
    renderer._mjr_context = new
    try:
        ctx.free()
    except Exception:  # noqa: BLE001 - an old context that cannot be freed is only leaked GL memory
        pass
    return on


# ---------------------------------------------------------------------- sim-state mutation provider
PROVIDER_KEY = "lighting"


def _pack(model) -> dict:
    snap = snapshot(model)
    return {} if snap is None else snap


def _unpack(model, data: dict):
    if data:
        restore(model, data)


def register_provider():
    """Register the ``lighting`` provider (``world/modules/terrain.py`` calls it when it composes an orchard
    world, and ``unregister_provider`` for any other world). Every process composes the same world, so every
    process holds the same provider set; the legacy world keeps an empty ``mutations`` block."""
    from atharv_nandi.core import sim_state
    sim_state.register_provider(PROVIDER_KEY, _pack, _unpack, version=1)


def unregister_provider():
    from atharv_nandi.core import sim_state
    sim_state.unregister_provider(PROVIDER_KEY)
