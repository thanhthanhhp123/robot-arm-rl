"""Train agent (SAC/PPO qua SB3) trên RobotArm2DEnv, hyperparameter từ YAML config.

Chạy: python train.py --config configs/sac.yaml
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from robot_arm.env import RobotArm2DEnv

ALGOS = {"sac": SAC, "ppo": PPO}


class DistanceLogger(BaseCallback):
    """Log khoảng cách trung bình ee↔target lên tensorboard mỗi log_freq bước."""

    def __init__(self, log_freq: int) -> None:
        super().__init__()
        self.log_freq = log_freq
        self._distances: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self._distances.append(float(info["distance"]))
        if self.n_calls % self.log_freq == 0 and self._distances:
            mean_dist = sum(self._distances) / len(self._distances)
            self.logger.record("rollout/mean_distance", mean_dist)
            self._distances.clear()
        return True


def load_config(path: str | Path) -> dict[str, Any]:
    """Đọc file YAML config."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(env_cfg: dict[str, Any], seed: int) -> Monitor:
    """Tạo env từ config, bọc Monitor để log episode reward/length."""
    env = RobotArm2DEnv(**env_cfg)
    env.reset(seed=seed)
    return Monitor(env)


def resolve_run_dir(train_cfg: dict[str, Any], run_dir_arg: str | None) -> Path:
    """Folder riêng cho mỗi session train: <output_root>/<run_name>_<timestamp>."""
    if run_dir_arg is not None:
        return Path(run_dir_arg)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(train_cfg["output_root"]) / f"{train_cfg['run_name']}_{stamp}"


def train(cfg: dict[str, Any], run_dir_arg: str | None = None) -> None:
    """Train agent theo config, log tensorboard + checkpoint vào run dir riêng."""
    seed: int = cfg["seed"]
    train_cfg: dict[str, Any] = cfg["train"]
    algo_name: str = cfg["algo"]
    algo_cls = ALGOS[algo_name]
    set_random_seed(seed)

    run_dir = resolve_run_dir(train_cfg, run_dir_arg)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Run dir: {run_dir}")

    env = make_env(cfg["env"], seed)
    model = algo_cls(
        env=env,
        seed=seed,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        **cfg[algo_name],
    )

    run_name: str = train_cfg["run_name"]
    callbacks = [
        CheckpointCallback(
            save_freq=train_cfg["checkpoint_freq"],
            save_path=str(checkpoint_dir),
            name_prefix=run_name,
        ),
        DistanceLogger(log_freq=train_cfg["distance_log_freq"]),
    ]

    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=callbacks,
        tb_log_name=run_name,
    )

    final_path = checkpoint_dir / f"{run_name}_final"
    model.save(final_path)
    print(f"Đã lưu model cuối cùng: {final_path}.zip")
    env.close()


def main() -> None:
    """Parse tham số dòng lệnh rồi train."""
    parser = argparse.ArgumentParser(description="Train SAC/PPO trên RobotArm2DEnv")
    parser.add_argument(
        "--config", type=str, default="configs/sac.yaml", help="Đường dẫn YAML config"
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Folder output của session (mặc định: tự tạo theo run_name + timestamp)",
    )
    args = parser.parse_args()
    train(load_config(args.config), args.run_dir)


if __name__ == "__main__":
    main()
