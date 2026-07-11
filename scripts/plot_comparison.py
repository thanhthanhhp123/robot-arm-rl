"""Vẽ so sánh learning curve giữa các run từ tensorboard event files.

Chạy: python scripts/plot_comparison.py --runs <run_dir1> <run_dir2> --out <png>
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Palette categorical (reference, đã validate CVD): blue, aqua, yellow, green, violet.
SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def load_scalar(tb_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Đọc 1 scalar tag từ folder tensorboard, trả (steps, values)."""
    event_dir = next(tb_dir.iterdir())  # tb/<run_name>_1/
    acc = EventAccumulator(str(event_dir))
    acc.Reload()
    events = acc.Scalars(tag)
    return np.array([e.step for e in events]), np.array([e.value for e in events])


def style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Trục kín đáo: grid mảnh, không viền thừa, chữ màu mực phụ."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("timesteps", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def plot_tag(
    ax: plt.Axes,
    run_dirs: list[Path],
    labels: list[str],
    tag: str,
    legend_loc: str = "upper right",
) -> None:
    """Vẽ 1 tag cho các run lên axes; nhận diện series qua legend + bảng metrics."""
    for run, label, color in zip(run_dirs, labels, SERIES_COLORS):
        steps, values = load_scalar(run / "tb", tag)
        ax.plot(steps, values, color=color, linewidth=2, label=label)
    ax.legend(loc=legend_loc, fontsize=9, frameon=False, labelcolor=INK)


def main() -> None:
    """Đọc các run dir, vẽ ep_rew_mean + mean_distance thành 1 hình PNG."""
    parser = argparse.ArgumentParser(description="So sánh learning curve")
    parser.add_argument("--runs", nargs="+", required=True, help="Các run dir")
    parser.add_argument("--labels", nargs="+", required=True, help="Tên series")
    parser.add_argument("--out", type=str, required=True, help="File PNG output")
    args = parser.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    plot_tag(
        axes[0], run_dirs, args.labels, "rollout/ep_rew_mean",
        legend_loc="lower right",
    )
    style_axis(axes[0], "Episode reward trung bình", "reward")
    plot_tag(axes[1], run_dirs, args.labels, "rollout/mean_distance")
    style_axis(axes[1], "Tracking error trung bình (ee ↔ target)", "distance")
    axes[1].set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
