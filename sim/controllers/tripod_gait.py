import math


class TripodGaitGenerator:
    """18-DOF alternating tripod gait generator for KisaanuNexusFarmDog6."""

    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self.vx = 0.0
        self.vy = 0.0
        self.is_held = False
        self.remaining_phase = 0.0
        self.gait_phase = 0.0
        self.leg_signs = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        self.tripod_phase = [0.0, math.pi, math.pi, 0.0, 0.0, math.pi]

    def set_command(self, vx: float, vy: float, hold: bool):
        if abs(vx) < 1e-3 and abs(vy) < 1e-3:
            self.is_held = False
            return
        self.vx = vx
        self.vy = vy
        self.is_held = hold
        if self.remaining_phase <= 0.05:
            self.remaining_phase = 2.0 * math.pi

    def stop(self):
        self.is_held = False
        self.remaining_phase = 0.0
        self.vx = 0.0
        self.vy = 0.0

    @property
    def is_active(self) -> bool:
        return self.is_held or (self.remaining_phase > 0.0)

    def step(self):
        active = self.is_active
        if active and math.hypot(self.vx, self.vy) > 1e-3:
            d_phase = 2.0 * math.pi * 1.8 * self.dt
            self.gait_phase = (self.gait_phase + d_phase) % (2.0 * math.pi)
            if not self.is_held:
                self.remaining_phase = max(0.0, self.remaining_phase - d_phase)
                if self.remaining_phase == 0.0:
                    self.vx = self.vy = 0.0
        else:
            self.gait_phase = 0.0

        targets = []
        for i in range(6):
            base_hip, base_thigh, base_calf = self.leg_signs[i] * 0.12, 0.80, -1.60
            if not active or math.hypot(self.vx, self.vy) < 1e-3:
                targets.extend([base_hip, base_thigh, base_calf])
                continue
            phase = self.gait_phase + self.tripod_phase[i]
            swing, lift = math.sin(phase), max(0.0, math.cos(phase))
            targets.extend([
                base_hip + 0.22 * self.vy * swing,
                base_thigh - 0.28 * self.vx * swing - 0.15 * lift,
                base_calf - 0.42 * lift,
            ])
        return targets
