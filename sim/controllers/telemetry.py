import math  # Import math for inverse cosine and radian-to-degree conversion
from typing import Sequence, Tuple  # Import type annotations for clarity


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
