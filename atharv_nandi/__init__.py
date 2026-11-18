"""Atharv Nandi (NDI): the Kisaanu orchard field robot simulator, one ROS 2 ament_python package (WPR).

    atharv_nandi.core    ROS-free logic (controllers, planners, perception, workflow runtime, HAL)
    atharv_nandi.sim     world builder (``sim.world``), per-step hooks (``sim.hooks``), MJCF models
    atharv_nandi.nodes   thin rclpy nodes (one console entry point each, ``nodes/entry.py``)
    atharv_nandi.web     the browser console assets (served by ``nodes/web_bridge.py``)
    atharv_nandi.paths   where config/, models/ and web/ live (source tree or ament share dir)
"""
__version__ = "0.1.0"
