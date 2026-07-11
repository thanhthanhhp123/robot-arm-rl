"""Generators quỹ đạo target cho RobotArm2DEnv: circle / figure-8 / spline.

Interface chung: thuộc tính `period` (giây/vòng) và `position(t)` -> (x, y).
Env random offset t0 mỗi episode để đa dạng điểm xuất phát.
"""

import numpy as np


class CircleTrajectory:
    """Target chạy đều trên đường tròn: center + radius·[cos(2πt/T), sin(2πt/T)]."""

    def __init__(
        self, center: tuple[float, float], radius: float, period: float
    ) -> None:
        """center: tâm (x, y); radius: bán kính; period: giây/vòng."""
        self.center = np.asarray(center, dtype=np.float32)
        self.radius = float(radius)
        self.period = float(period)

    def position(self, t: float) -> np.ndarray:
        """Vị trí target tại thời điểm t (giây)."""
        angle = 2.0 * np.pi * t / self.period
        offset = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        return self.center + self.radius * offset


class FigureEightTrajectory:
    """Hình số 8 (Lissajous 1:2): center + [a·sin(θ), b·sin(2θ)], θ = 2πt/T."""

    def __init__(
        self,
        center: tuple[float, float],
        width: float,
        height: float,
        period: float,
    ) -> None:
        """width: biên độ x (a); height: biên độ y (b); period: giây/vòng."""
        self.center = np.asarray(center, dtype=np.float32)
        self.width = float(width)
        self.height = float(height)
        self.period = float(period)

    def position(self, t: float) -> np.ndarray:
        """Vị trí target tại thời điểm t (giây)."""
        angle = 2.0 * np.pi * t / self.period
        offset = np.array(
            [self.width * np.sin(angle), self.height * np.sin(2.0 * angle)],
            dtype=np.float32,
        )
        return self.center + offset


class SplineTrajectory:
    """Đường cong khép kín Catmull-Rom đi qua các điểm tự vẽ (control points).

    Đi hết 1 vòng qua N điểm trong `period` giây; đoạn thứ i chiếm period/N.
    """

    def __init__(self, points: list[list[float]], period: float) -> None:
        """points: danh sách [x, y] (>= 4 điểm); period: giây/vòng."""
        self.points = np.asarray(points, dtype=np.float32)
        if len(self.points) < 4:
            raise ValueError("SplineTrajectory cần >= 4 control points")
        self.period = float(period)

    def position(self, t: float) -> np.ndarray:
        """Vị trí target tại thời điểm t (giây), nội suy Catmull-Rom."""
        n = len(self.points)
        u = (t % self.period) / self.period * n
        i = int(u) % n
        s = u - int(u)
        p0 = self.points[(i - 1) % n]
        p1 = self.points[i]
        p2 = self.points[(i + 1) % n]
        p3 = self.points[(i + 2) % n]
        return (
            0.5
            * (
                2.0 * p1
                + (-p0 + p2) * s
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * s**2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * s**3
            )
        ).astype(np.float32)


TRAJECTORY_CLASSES = {
    "circle": CircleTrajectory,
    "figure8": FigureEightTrajectory,
    "spline": SplineTrajectory,
}


def make_trajectory(name: str, params: dict) -> object | None:
    """Tạo trajectory theo tên; "fixed" -> None (target đứng yên)."""
    if name == "fixed":
        return None
    if name not in TRAJECTORY_CLASSES:
        raise ValueError(f"trajectory không hợp lệ: {name!r}")
    return TRAJECTORY_CLASSES[name](**params)
