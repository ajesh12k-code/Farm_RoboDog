"""Tool runtime packages: WP2 (registry, tool changer, grasp sense, cut and twist logic), WP3 (kinematic transfer).

Every module here is ROS-free. The pure-logic modules (``cut_logic``, ``twist_logic``, ``grasp_sense``, the
``tool_changer`` state machine) need neither ROS nor MuJoCo; ``registry.ToolHandles`` and the
``tool_changer.MjToolPlant`` glue read a compiled ``MjModel``.
"""
