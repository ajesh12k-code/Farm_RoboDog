"""Active-time clock (WP34; WORKFLOWS-SPEC A.2 "Timeouts count ACTIVE time only", A.4 pause).

A workflow timeout counts only the time its tree is actually ticked, not wall or sim time: the prototype showed
that otherwise a long pause makes a step time out the moment it resumes. Each tree the engine ticks (main tree,
``on_pause`` / ``on_abort`` finaliser, ``resume_check``) has its own ``ActiveClock``; the engine advances it by
the tick period only while that tree is ticked, and ``freeze()`` makes ``advance`` a no-op (pause).

``total`` is the run time including pauses (sim time of the driver, or ticks x dt in a unit test): the pause
length that decides whether resume needs an operator ``ack`` (A.4 step 3) is measured on it.
"""


class ActiveClock:
    __slots__ = ("active", "total", "frozen")

    def __init__(self, active: float = 0.0, total: float = 0.0):
        self.active = float(active)
        self.total = float(total)
        self.frozen = False

    def now(self) -> float:
        """Active seconds so far (what timeouts and backoffs compare against)."""
        return self.active

    def advance(self, dt: float, active: bool = True) -> None:
        """One tick of ``dt`` seconds: ``total`` always moves, ``active`` only when not frozen and ``active``."""
        if dt < 0:
            raise ValueError("clock cannot run backwards")
        self.total += dt
        if active and not self.frozen:
            self.active += dt

    def freeze(self) -> None:
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False

    def to_dict(self) -> dict:
        return {"active": self.active, "total": self.total}

    @classmethod
    def from_dict(cls, d: dict) -> "ActiveClock":
        return cls(float(d.get("active", 0.0)), float(d.get("total", 0.0)))


class Stopwatch:
    """Elapsed active time since ``start`` on a clock; ``elapsed`` survives a freeze (it does not grow)."""
    __slots__ = ("clock", "t0")

    def __init__(self, clock: ActiveClock):
        self.clock = clock
        self.t0 = clock.now()

    def restart(self) -> None:
        self.t0 = self.clock.now()

    def elapsed(self) -> float:
        return self.clock.now() - self.t0

    def expired(self, limit_s) -> bool:
        return limit_s is not None and self.elapsed() >= float(limit_s) - 1e-9
