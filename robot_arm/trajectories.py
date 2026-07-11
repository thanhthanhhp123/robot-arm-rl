"""Generators quỹ đạo target cho RobotArm2DEnv (Slice 4: đường tròn)."""

import numpy as np


class CircleTrajectory:
    """Target chạy đều trên đường tròn, tham số hóa theo thời gian t (giây).

    position(t) = center + radius * [cos(phase + 2πt/period), sin(...)]
    """

    def __init__(
        self, center: tuple[float, float], radius: float, period: float
    ) -> None:
        """Khởi tạo đường tròn.

        Args:
            center: Tâm đường tròn (x, y).
            radius: Bán kính.
            period: Thời gian đi hết 1 vòng (giây).
        """
        self.center = np.asarray(center, dtype=np.float32)
        self.radius = float(radius)
        self.period = float(period)

    def position(self, t: float, phase: float = 0.0) -> np.ndarray:
        """Vị trí target tại thời điểm t, với pha ban đầu phase (rad)."""
        angle = phase + 2.0 * np.pi * t / self.period
        offset = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        return self.center + self.radius * offset
