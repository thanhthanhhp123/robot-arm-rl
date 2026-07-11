"""Gymnasium environment: 2-link planar robot arm reaching một điểm cố định."""

from typing import Any

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # headless, không mở cửa sổ

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

from robot_arm.kinematics import end_effector_position, forward_kinematics
from robot_arm.trajectories import CircleTrajectory


class RobotArm2DEnv(gym.Env):
    """2-link planar robot arm học bám target (cố định hoặc chạy theo quỹ đạo).

    Observation (float32, shape (10,)):
        [cos θ1, sin θ1, cos θ2, sin θ2, θ1_dot, θ2_dot, ex, ey, ex-px, ey-py]

    Action (float32, shape (2,), range [-1, 1]):
        Kinematic control: Δθ_i = action_i * max_angular_step.

    Reward: -||p_ee - target|| (chỉ term khoảng cách, chưa shaping thêm).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        l1: float = 1.0,
        l2: float = 1.0,
        max_steps: int = 200,
        max_angular_step: float = 0.1,
        dt: float = 0.05,
        target_bounds: tuple[float, float] = (0.5, 1.9),
        trajectory: str = "fixed",
        circle_center: tuple[float, float] = (1.0, 0.0),
        circle_radius: float = 0.5,
        circle_period: float = 10.0,
        render_mode: str | None = None,
    ) -> None:
        """Khởi tạo môi trường.

        Args:
            l1: Độ dài link 1.
            l2: Độ dài link 2.
            max_steps: Số bước tối đa mỗi episode trước khi truncate.
            max_angular_step: Bước góc tối đa mỗi step (rad), action=1 -> Δθ này.
            dt: Thời gian mỗi step (s), dùng để tính θ_dot và tham số t quỹ đạo.
            target_bounds: (min, max) bán kính lấy mẫu target quanh base,
                phải nằm trong workspace [|l1-l2|, l1+l2] (chế độ "fixed").
            trajectory: "fixed" (target đứng yên) hoặc "circle" (chạy đường tròn).
            circle_center: Tâm đường tròn quỹ đạo target.
            circle_radius: Bán kính đường tròn.
            circle_period: Chu kỳ 1 vòng (giây), t = step_count * dt.
            render_mode: Chế độ render (None hoặc "rgb_array").
        """
        super().__init__()
        if trajectory not in ("fixed", "circle"):
            raise ValueError(f"trajectory không hợp lệ: {trajectory!r}")
        self.l1 = l1
        self.l2 = l2
        self.max_steps = max_steps
        self.max_angular_step = max_angular_step
        self.dt = dt
        self.target_bounds = target_bounds
        self.trajectory = trajectory
        self._circle = (
            CircleTrajectory(circle_center, circle_radius, circle_period)
            if trajectory == "circle"
            else None
        )
        self._phase: float = 0.0
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        max_reach = l1 + l2
        max_theta_dot = max_angular_step / dt
        obs_low = np.array(
            [-1.0, -1.0, -1.0, -1.0, -max_theta_dot, -max_theta_dot,
             -max_reach, -max_reach, -2 * max_reach, -2 * max_reach],
            dtype=np.float32,
        )
        obs_high = -obs_low
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.theta: np.ndarray = np.zeros(2, dtype=np.float32)
        self.theta_dot: np.ndarray = np.zeros(2, dtype=np.float32)
        self.target: np.ndarray = np.zeros(2, dtype=np.float32)
        self.step_count: int = 0

        self._fig = None
        self._ax = None

    def _sample_target(self) -> np.ndarray:
        """Lấy mẫu target ngẫu nhiên trong workspace khả thi của tay máy."""
        radius = self.np_random.uniform(*self.target_bounds)
        angle = self.np_random.uniform(-np.pi, np.pi)
        return np.array(
            [radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        """Xây observation từ state hiện tại."""
        ee = end_effector_position(self.theta[0], self.theta[1], self.l1, self.l2)
        error = self.target - ee
        return np.array(
            [
                np.cos(self.theta[0]), np.sin(self.theta[0]),
                np.cos(self.theta[1]), np.sin(self.theta[1]),
                self.theta_dot[0], self.theta_dot[1],
                self.target[0], self.target[1],
                error[0], error[1],
            ],
            dtype=np.float32,
        )

    def _get_info(self, ee: np.ndarray, dist: float) -> dict[str, Any]:
        """Info dict để debug từng thành phần reward/state."""
        return {
            "distance": dist,
            "end_effector": ee,
            "target": self.target.copy(),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset môi trường: random góc khớp về 0, lấy target mới."""
        super().reset(seed=seed)
        self.theta = np.zeros(2, dtype=np.float32)
        self.theta_dot = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        if self._circle is not None:
            self._phase = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            self.target = self._circle.position(0.0, self._phase)
        else:
            self.target = self._sample_target()

        ee = end_effector_position(self.theta[0], self.theta[1], self.l1, self.l2)
        dist = float(np.linalg.norm(self.target - ee))
        return self._get_obs(), self._get_info(ee, dist)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Áp Δθ = action * max_angular_step, trả 5-tuple Gymnasium."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        delta_theta = action * self.max_angular_step

        prev_theta = self.theta.copy()
        self.theta = self.theta + delta_theta
        self.theta_dot = (self.theta - prev_theta) / self.dt
        self.step_count += 1

        if self._circle is not None:
            t = self.step_count * self.dt
            self.target = self._circle.position(t, self._phase)

        ee = end_effector_position(self.theta[0], self.theta[1], self.l1, self.l2)
        dist = float(np.linalg.norm(self.target - ee))
        reward = -dist

        terminated = False
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info(ee, dist)

    def render(self) -> np.ndarray | None:
        """Vẽ frame hiện tại (2 link, end-effector, target) khi render_mode='rgb_array'."""
        if self.render_mode != "rgb_array":
            return None

        from robot_arm.render import draw_frame, figure_to_array

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(5, 5))

        base, elbow, ee = forward_kinematics(self.theta[0], self.theta[1], self.l1, self.l2)
        draw_frame(self._ax, base, elbow, ee, self.target, self.l1 + self.l2)
        return figure_to_array(self._fig)

    def close(self) -> None:
        """Đóng figure matplotlib nếu đã tạo."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
