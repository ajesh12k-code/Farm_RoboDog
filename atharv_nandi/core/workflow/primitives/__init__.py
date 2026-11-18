"""Workflow primitives (WP34; WORKFLOWS-SPEC A.3).

``registry.py`` (WP34) is the catalogue v1 and the adapter binding point; ``stubs.py`` (WP34) binds a stub for
every primitive. The adapter files ``motion.py``, ``mast.py``, ``sensing.py``, ``records.py``, ``arm.py``,
``assist.py``, ``conditions.py`` are WP35's and ``dock.py`` is WP38's: thin wrappers over existing controllers
that call ``registry.bind(name, factory)``.
"""
