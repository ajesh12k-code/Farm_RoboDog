"""WP8 task planners per crop (ORCHARD-MVP-SPEC 2.2, 2.4, 3.4, 5.3; MISSION-PRESETS 5, 6).

Pure state machines (no MuJoCo, no ROS): ``clip_pedicel`` (T2: mango one round, guava clip), ``twist_pick`` (T4:
papaya with ``jaw_clearance``, guava twist), ``prune_cut`` (T5 prune jobs), ``banana_tidy`` (T5 dead-blade tidy +
banana MARK planner); ``tree_task`` builds a tree's targets and MARKs from WP23 truth and the crop profile; ``common``
holds records (MarkRecord ``nandi.mark/1``, HarvestRecord) and the SM base. ``sim_executor`` runs them on a
``SimDriver`` with the WP2 tools, the WP3 transfer and the WPCAP planner (MuJoCo, ROS-free). Retry / skip / halt:
``controllers.task_retry``. Contract: WP8.
"""
