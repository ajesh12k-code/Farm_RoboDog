"""WPCAP arm motion planning: MuJoCo collision scene + OMPL (or pure-Python RRT-Connect) joint-space planner.

    from atharv_nandi.core.planning import CollisionScene, ArmPlanner
    planner = ArmPlanner(model)                 # backend "auto": OMPL if importable, else "rrtconnect"
    res = planner.plan(data.qpos, goal_q)       # PlanResult: ok, path (N x 5), trajectory, numbers
"""
from atharv_nandi.core.planning.collision_scene import CollisionScene, ARM_JOINTS, DEFAULT_MARGINS  # noqa: F401
from atharv_nandi.core.planning.ompl_planner import (  # noqa: F401
    ArmPlanner, PlanResult, Trajectory, ompl_available, time_parameterize,
)
