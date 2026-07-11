"""Render 2-link robot arm bằng matplotlib: vẽ frame, ghi episode, xuất GIF."""

from typing import Callable

import matplotlib

matplotlib.use("Agg")  # headless, không mở cửa sổ

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from robot_arm.kinematics import forward_kinematics


def draw_frame(
    ax: plt.Axes,
    base: np.ndarray,
    elbow: np.ndarray,
    ee: np.ndarray,
    target: np.ndarray,
    reach: float,
    target_trail: np.ndarray | None = None,
    ee_trail: np.ndarray | None = None,
) -> None:
    """Vẽ 1 frame (2 link, khớp, end-effector, target) lên axes có sẵn.

    target_trail / ee_trail: mảng (N, 2) các vị trí đã đi qua, để overlay
    quỹ đạo target vs quỹ đạo end-effector thực tế.
    """
    ax.clear()
    margin = reach * 1.15
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if target_trail is not None and len(target_trail) > 1:
        ax.plot(
            target_trail[:, 0], target_trail[:, 1], "--", color="crimson",
            linewidth=1.2, alpha=0.6, zorder=1, label="quỹ đạo target",
        )
    if ee_trail is not None and len(ee_trail) > 1:
        ax.plot(
            ee_trail[:, 0], ee_trail[:, 1], "-", color="darkorange",
            linewidth=1.2, alpha=0.6, zorder=1, label="quỹ đạo ee",
        )

    xs, ys = [base[0], elbow[0], ee[0]], [base[1], elbow[1], ee[1]]
    ax.plot(xs, ys, "-o", color="steelblue", linewidth=3, markersize=6, zorder=2)
    ax.plot(*base, "ks", markersize=8, zorder=3)
    ax.plot(*ee, "o", color="darkorange", markersize=10, zorder=3, label="end-effector")
    ax.plot(
        *target, "x", color="crimson", markersize=12, markeredgewidth=3,
        zorder=3, label="target",
    )
    ax.legend(loc="upper right", fontsize=8)


def figure_to_array(fig: plt.Figure) -> np.ndarray:
    """Chuyển canvas matplotlib hiện tại thành ảnh RGB (H, W, 3) uint8."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[:, :, :3].copy()


def record_episode(
    env: gym.Env,
    policy: Callable[[np.ndarray], np.ndarray] | None = None,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Chạy 1 episode (random policy nếu không truyền policy), trả list frame RGB.

    `env` phải được tạo với `render_mode="rgb_array"`.
    """
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]

    terminated = truncated = False
    while not (terminated or truncated):
        action = policy(obs) if policy is not None else env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
    return frames


def save_gif(frames: list[np.ndarray], path: str, fps: int = 20) -> None:
    """Lưu list frame RGB thành file GIF."""
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )


if __name__ == "__main__":
    from robot_arm.env import RobotArm2DEnv

    env = RobotArm2DEnv(render_mode="rgb_array")
    frames = record_episode(env, seed=0)
    env.close()

    out_path = "outputs/random_rollout.gif"
    save_gif(frames, out_path, fps=20)
    print(f"Saved {len(frames)} frames -> {out_path}")
