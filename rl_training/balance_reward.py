import numpy as np  # Import NumPy for vector norms, exponentials, and squared errors


def compute_arm_balance_reward(
    x_before: float,
    x_after: float,
    dt: float,
    qpos: np.ndarray,
    qvel: np.ndarray,
    arm_qpos_err: np.ndarray,
    arm_qvel: np.ndarray,
    leg_action: np.ndarray,
    last_action: np.ndarray,
    full_action: np.ndarray,
    tip_world_pos: np.ndarray,
    trunk_world_pos: np.ndarray,
    holding_item: bool = False,
):
    """
    Reward forward walking with the dorsal arm tucked cleanly at rest during normal
    locomotion, deploying ONLY during steep climbs/pushes/slips (if empty) and
    retracting immediately once stability is restored.
    """
    # Compute forward walking velocity (m/s) along world X
    forward_vel = (x_after - x_before) / dt
    # Target cruising speed (m/s) over farm terrain
    target_vel = 0.45

    # Low survival bonus so the robot must walk forward rather than stand still
    alive_bonus = 0.25
    # Linear forward velocity progress (capped at 0.75 m/s)
    vel_progress = 3.5 * min(forward_vel, 0.75)
    # Gaussian speed-tracking bonus centered on 0.45 m/s
    vel_tracking_bonus = 1.5 * float(np.exp(-4.0 * ((forward_vel - target_vel) ** 2)))
    # Penalty if the robot stops walking forward
    standstill_penalty = 0.50 if forward_vel < 0.08 else 0.0

    # Extract torso tilt proxies from quaternion [w, x, y, z] and angular rates
    roll_proxy = float(qpos[4])
    pitch_proxy = float(qpos[5])
    tilt_mag = float(np.hypot(roll_proxy, pitch_proxy))
    ang_vel_mag = float(np.hypot(qvel[3], qvel[4]))

    # Penalize torso roll/pitch tilt, lateral drift, and yaw rotation
    tilt_penalty = 2.8 * (roll_proxy ** 2 + pitch_proxy ** 2)
    lateral_penalty = 0.30 * abs(float(qvel[1]))
    yaw_penalty = 0.20 * abs(float(qvel[5]))

    # Smoothness penalty across all 23 actions and leg energy penalty
    action_rate_cost = 0.015 * float(np.sum(np.square(full_action - last_action)))
    leg_energy_cost = 0.002 * float(np.sum(np.square(leg_action)))

    # --- Conditional Dorsal Arm Behavior: Rest vs. Emergency Stabilization ---
    arm_support_bonus = 0.0
    arm_rest_penalty = 0.0

    # Define difficulty/instability threshold (steep climb, slip, or external push)
    is_unstable = (tilt_mag > 0.11) or (ang_vel_mag > 0.85)

    if not holding_item and is_unstable:
        # EMERGENCY STABILIZATION MODE: reward counterweight extension & ground bracing
        tip_rel_x = float(tip_world_pos[0] - trunk_world_pos[0])
        tip_rel_y = float(tip_world_pos[1] - trunk_world_pos[1])

        counter_y = max(0.0, roll_proxy * tip_rel_y)
        counter_x = max(0.0, -pitch_proxy * tip_rel_x)
        arm_support_bonus += 4.0 * (counter_y + counter_x)

        # Extra bonus if the gripper tip touches the ground (z <= 0.06 m) to brace a fall
        if tip_world_pos[2] <= 0.06:
            arm_support_bonus += 1.0
    else:
        # NORMAL WALKING OR HOLDING ITEM: strongly enforce folded rest posture & zero jitter
        arm_rest_penalty = (
            0.45 * float(np.sum(np.square(arm_qpos_err)))
            + 0.04 * float(np.sum(np.square(arm_qvel)))
        )

    # Combine all positive rewards and energy/posture penalties
    reward = float(
        vel_progress
        + vel_tracking_bonus
        + alive_bonus
        + arm_support_bonus
        - standstill_penalty
        - tilt_penalty
        - lateral_penalty
        - yaw_penalty
        - action_rate_cost
        - leg_energy_cost
        - arm_rest_penalty
    )

    # Terminate episode if torso drops below 0.20 m or flips excessively
    fallen = bool(qpos[2] < 0.20 or qpos[3] < 0.55)

    info = {
        "forward_vel": float(forward_vel),
        "x_pos": float(x_after),
        "tilt_mag": tilt_mag,
        "arm_support_bonus": float(arm_support_bonus),
        "arm_rest_err": float(np.linalg.norm(arm_qpos_err)),
        "tip_z": float(tip_world_pos[2]),
    }
    return reward, fallen, info
