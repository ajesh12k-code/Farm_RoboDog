"""One file per work package that adds world content (WP0 compose registry).

Each ``<name>.py`` here calls ``world.compose.register_module(name, order, attach_fn, version=...)`` at
import time; ``compose.discover()`` imports every file in this directory. Files starting with ``_`` are
skipped. Register sim-state mutation providers (``controllers.sim_state.register_provider``) in the same
file, so every process that builds the world (physics, perception, web) also knows how to apply them.
"""
