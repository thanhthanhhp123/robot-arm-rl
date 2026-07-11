"""Đánh giá agent đã train: tracking error, success rate, video overlay.

Chạy: python eval.py --run-dir outputs/runs/<run>  (đọc config.yaml trong run dir)
"""

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from robot_arm.env import RobotArm2DEnv
from robot_arm.kinematics import forward_kinematics
from robot_arm.render import draw_frame, figure_to_array, save_gif

ALGOS = {"sac": SAC, "ppo": PPO}


def rollout(
    env: RobotArm2DEnv, model: BaseAlgorithm, seed: int
) -> dict[str, np.ndarray]:
    """Chạy 1 episode deterministic, trả quỹ đạo ee/target, distance, action."""
    obs, info = env.reset(seed=seed)
    ee_path, target_path, dists, action_norms = [], [], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        ee_path.append(info["end_effector"])
        target_path.append(info["target"])
        dists.append(info["distance"])
        action_norms.append(float(np.linalg.norm(action)))
    return {
        "ee_path": np.array(ee_path),
        "target_path": np.array(target_path),
        "dists": np.array(dists),
        "action_norms": np.array(action_norms),
    }


def compute_metrics(
    env: RobotArm2DEnv, model: BaseAlgorithm, n_episodes: int, threshold: float
) -> dict[str, float]:
    """Metrics trung bình trên n_episodes: mean/max error, success rate, smoothness."""
    all_dists, all_action_norms = [], []
    for ep in range(n_episodes):
        result = rollout(env, model, seed=ep)
        all_dists.append(result["dists"])
        all_action_norms.append(result["action_norms"])
    dists = np.concatenate(all_dists)
    action_norms = np.concatenate(all_action_norms)
    return {
        "mean_tracking_error": float(dists.mean()),
        "max_tracking_error": float(dists.max()),
        "success_rate": float((dists < threshold).mean()),
        "mean_action_norm": float(action_norms.mean()),
        "n_episodes": n_episodes,
        "success_threshold": threshold,
    }


def make_overlay_video(
    env: RobotArm2DEnv, model: BaseAlgorithm, path: Path, seed: int, fps: int
) -> None:
    """Xuất GIF 1 episode với overlay quỹ đạo target (đỏ) vs ee thực (cam)."""
    obs, info = env.reset(seed=seed)
    fig, ax = plt.subplots(figsize=(5, 5))
    frames = []
    ee_trail = [info["end_effector"]]
    target_trail = [info["target"]]

    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        ee_trail.append(info["end_effector"])
        target_trail.append(info["target"])
        base, elbow, ee = forward_kinematics(
            env.theta[0], env.theta[1], env.l1, env.l2
        )
        draw_frame(
            ax, base, elbow, ee, info["target"], env.l1 + env.l2,
            target_trail=np.array(target_trail), ee_trail=np.array(ee_trail),
        )
        frames.append(figure_to_array(fig))
    plt.close(fig)
    save_gif(frames, str(path), fps=fps)


def main() -> None:
    """Load model final từ run dir, in metrics và xuất video overlay."""
    parser = argparse.ArgumentParser(description="Đánh giá agent đã train")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--video-seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "config.yaml", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    eval_cfg = cfg["eval"]

    final_model = run_dir / "checkpoints" / f"{cfg['train']['run_name']}_final"
    model = ALGOS[cfg["algo"]].load(final_model, device="cpu")
    env = RobotArm2DEnv(**cfg["env"])

    metrics = compute_metrics(
        env, model, eval_cfg["n_episodes"], eval_cfg["success_threshold"]
    )
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    with open(eval_dir / "metrics.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    video_path = eval_dir / "overlay.gif"
    make_overlay_video(env, model, video_path, args.video_seed, args.fps)
    print(f"Video overlay: {video_path}")
    env.close()


if __name__ == "__main__":
    main()
