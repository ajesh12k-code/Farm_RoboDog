import json  # Import json for the std_msgs/String status topics
import math  # Import math for inverse cosine and radian-to-degree conversion
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple  # Import type annotations for clarity


def parse_qpos_telemetry(qpos: Sequence[float]) -> Tuple[float, float, float]:
    """Extract (forward_x_m, height_z_m, tilt_deg) from a MuJoCo qpos vector."""
    # Return zeros if the state vector does not contain full 7-DOF base pose
    if len(qpos) < 7:
        return 0.0, 0.0, 0.0

    # Read chassis forward position (X) and vertical height above ground (Z)
    x, z = float(qpos[0]), float(qpos[2])
    # Unpack torso orientation quaternion [w, qx, qy, qz]
    _, qx, qy, _ = qpos[3:7]

    # Compute cosine of total tilt angle relative to world vertical axis (Z)
    cos_tilt = max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))
    # Convert tilt angle from radians to degrees
    tilt_deg = math.degrees(math.acos(cos_tilt))
    return x, z, tilt_deg


def format_telemetry(telem: Tuple[float, float, float]) -> Tuple[str, str]:
    """Return formatted status bar text and warning color for (x, z, tilt_deg)."""
    x, z, tilt = telem
    # Select red (>25° danger), amber (>12° slope/push), or cyan (level)
    color = "#f87171" if tilt > 25.0 else ("#fbbf24" if tilt > 12.0 else "#38bdf8")
    return f"Dist: {x:+.2f} m  |  Height: {z:.2f} m  |  Tilt: {tilt:.1f}°", color


# ---- JSON status topics (std_msgs/String) -----------------------------------------------------
# Each parser accepts the raw message text (or None before the first message) and always returns a
# complete dict with safe defaults. "ok" is False when nothing valid has been received, so the GUI
# can show "waiting" instead of pretending the defaults are live values.
OK_GREEN, WARN_AMBER, BAD_RED, IDLE_GREY = "#34d399", "#fbbf24", "#f87171", "#94a3b8"

LOCO_DEFAULTS = {"mode": "tripod", "policy_loaded": False, "policy_path": None,
                 "gait_params": [1.0, 1.0, 1.8, 1.0], "walking": False}
HARVEST_DEFAULTS = {"mode": "rl_balance", "phase": "idle", "target": None, "last_result": ""}
GAME_DEFAULTS = {"crops_total": 0, "in_basket": 0, "in_basket_names": [], "falls": 0,
                 "fallen": False, "elapsed_s": 0.0, "rtf": 0.0}


def _load_obj(text) -> Optional[dict]:
    """json.loads that returns None unless the payload is a non-empty str holding a JSON object."""
    if not isinstance(text, str) or not text:
        return None
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _num(v, default: float) -> float:
    """Finite float or default (bool is rejected: True is not a number here)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return default
    return float(v)


def _int(v, default: int) -> int:
    return int(_num(v, float(default)))


def _str(v, default):
    return v if isinstance(v, str) else default


def _bool(v, default: bool) -> bool:
    return v if isinstance(v, bool) else default


def parse_locomotion_status(text) -> Dict[str, Any]:
    """Parse /nandi/locomotion_status JSON into a normalized dict (+ "ok")."""
    obj = _load_obj(text)
    out = dict(LOCO_DEFAULTS, gait_params=list(LOCO_DEFAULTS["gait_params"]), ok=obj is not None)
    if obj is None:
        return out
    mode = obj.get("mode")
    out["mode"] = mode if mode in ("tripod", "rl") else LOCO_DEFAULTS["mode"]
    out["policy_loaded"] = _bool(obj.get("policy_loaded"), False)
    out["policy_path"] = _str(obj.get("policy_path"), None)
    out["walking"] = _bool(obj.get("walking"), False)
    gp = obj.get("gait_params")
    if isinstance(gp, list) and len(gp) == 4:
        out["gait_params"] = [_num(v, d) for v, d in zip(gp, LOCO_DEFAULTS["gait_params"])]
    return out


def parse_harvest_status(text) -> Dict[str, Any]:
    """Parse /nandi/harvest_status JSON into a normalized dict (+ "ok")."""
    obj = _load_obj(text)
    out = dict(HARVEST_DEFAULTS, ok=obj is not None)
    if obj is None:
        return out
    mode = obj.get("mode")
    out["mode"] = mode if mode in ("rl_balance", "ik_harvest", "manual") else HARVEST_DEFAULTS["mode"]
    out["phase"] = _str(obj.get("phase"), HARVEST_DEFAULTS["phase"])
    out["target"] = _str(obj.get("target"), None)
    out["last_result"] = _str(obj.get("last_result"), "")
    return out


def parse_game_status(text) -> Dict[str, Any]:
    """Parse /nandi/game_status JSON into a normalized dict (+ "ok")."""
    obj = _load_obj(text)
    out = dict(GAME_DEFAULTS, in_basket_names=[], ok=obj is not None)
    if obj is None:
        return out
    out["crops_total"] = max(0, _int(obj.get("crops_total"), 0))
    out["in_basket"] = max(0, _int(obj.get("in_basket"), 0))
    names = obj.get("in_basket_names")
    if isinstance(names, list):
        out["in_basket_names"] = [n for n in names if isinstance(n, str)]
    out["falls"] = max(0, _int(obj.get("falls"), 0))
    out["fallen"] = _bool(obj.get("fallen"), False)
    out["elapsed_s"] = max(0.0, _num(obj.get("elapsed_s"), 0.0))
    out["rtf"] = max(0.0, _num(obj.get("rtf"), 0.0))
    return out


class ModeView(NamedTuple):
    """How the locomotion-mode label and gait toggle button should look."""
    label: str
    color: str
    toggle_enabled: bool


def format_mode(loco: Dict[str, Any]) -> ModeView:
    """Truthful mode label from locomotion_status; the toggle is only live when a policy is loaded."""
    if not loco.get("ok"):
        return ModeView("Locomotion: waiting for status...", IDLE_GREY, False)
    walk = " - walking" if loco.get("walking") else ""
    if not loco.get("policy_loaded"):
        return ModeView(f"Tripod Gait (no RL policy){walk}", WARN_AMBER, False)
    if loco.get("mode") == "rl":
        return ModeView(f"Mode: RL Balance Policy{walk}", OK_GREEN, True)
    return ModeView(f"Mode: Tripod Gait{walk}", OK_GREEN, True)


def format_clock(seconds: float) -> str:
    """mm:ss (minutes keep counting past 59)."""
    total = int(max(0.0, seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def format_game(game: Dict[str, Any]) -> Tuple[str, str]:
    """Score line 'Crops k/N | Falls f | mm:ss | RTF r' and its color (red while fallen)."""
    if not game.get("ok"):
        return "Score: waiting for sim...", IDLE_GREY
    text = (f"Crops {game['in_basket']}/{game['crops_total']} | Falls {game['falls']} | "
            f"{format_clock(game['elapsed_s'])} | RTF {game['rtf']:.2f}")
    if game.get("fallen"):
        return text + " | FALLEN", BAD_RED
    done = game["crops_total"] > 0 and game["in_basket"] >= game["crops_total"]
    return text, (OK_GREEN if done else "#e6edf3")


_ARM_MODE_TEXT = {"rl_balance": "RL balance", "ik_harvest": "IK harvest", "manual": "Manual"}


def format_harvest(h: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return (arm line, last-result line, color) for the harvest status panel."""
    if not h.get("ok"):
        return "Arm: waiting for harvest node...", "", IDLE_GREY
    line = f"Arm: {_ARM_MODE_TEXT.get(h['mode'], h['mode'])}  |  {h['phase']}"
    if h.get("target"):
        line += f"  |  {h['target']}"
    color = WARN_AMBER if h["mode"] == "ik_harvest" else (IDLE_GREY if h["mode"] == "rl_balance" else "#38bdf8")
    last = f"Last: {h['last_result']}" if h.get("last_result") else ""
    return line, last, color


def shorten(text: str, limit: int = 52) -> str:
    """Trim a status string to fit the narrow panel."""
    text = text or ""
    return text if len(text) <= limit else text[: limit - 3] + "..."
