import numpy as np  # Import NumPy for fast vector norms and squared-sum penalties


def compute_locomotion_reward(
    x_before: float,
    x_after: float,
    dt: float,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    last_action: np.ndarray,
):
    """Calculate scalar reward, fall termination flag, and diagnostic info."""
    # Compute forward velocity (m/s) along the world X-axis over this control step
    forward_vel = (x_after - x_before) / dt
    # Constant survival bonus per step to encourage staying upright
    alive_bonus = 0.6
    # Penalize abrupt action jumps between consecutive steps for smooth motor commands
    action_rate_cost = 0.015 * float(np.sum(np.square(action - last_action)))
    # Small penalty on total action magnitude to discourage excessive joint excursions
    energy_cost = 0.002 * float(np.sum(np.square(action)))
    # Penalize sideways drift velocity (world Y-axis) so the hexapod walks straight
    lateral_penalty = 0.35 * abs(float(qvel[1]))
    # Penalize spinning/yaw rate around the vertical Z-axis
    yaw_rate_penalty = 0.20 * abs(float(qvel[5]))
    # Penalize roll (qpos[4]) and pitch (qpos[5]) tilt so the dorsal basket stays level
    tilt_penalty = 2.0 * float(qpos[4] ** 2 + qpos[5] ** 2)

    # Combine positive forward progress with stability and smoothness penalties
    reward = float(
        2.5 * forward_vel
        + alive_bonus
        - action_rate_cost
        - energy_cost
        - lateral_penalty
        - yaw_rate_penalty
        - tilt_penalty
    )

    # Terminate early if torso height drops below 0.21 m or upright quaternion w < 0.60
    fallen = bool(qpos[2] < 0.21 or qpos[3] < 0.60)
    # Return scalar reward, boolean termination flag, and telemetry dictionary
    return reward, fallen, {"forward_vel": float(forward_vel), "x_pos": float(x_after)}
