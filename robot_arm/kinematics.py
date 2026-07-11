"""Forward kinematics cho 2-link planar robot arm."""

import numpy as np


def forward_kinematics(
    theta1: float, theta2: float, l1: float, l2: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tính vị trí base, khuỷu tay (elbow) và end-effector.

    theta1 đo từ trục x dương tại base. theta2 đo tương đối so với link 1.

    Args:
        theta1: Góc khớp 1 (rad).
        theta2: Góc khớp 2 (rad), tương đối so với link 1.
        l1: Độ dài link 1.
        l2: Độ dài link 2.

    Returns:
        Tuple (base_pos, elbow_pos, ee_pos), mỗi phần tử là np.ndarray shape (2,).
    """
    base_pos = np.array([0.0, 0.0], dtype=np.float32)
    elbow_pos = base_pos + l1 * np.array(
        [np.cos(theta1), np.sin(theta1)], dtype=np.float32
    )
    ee_pos = elbow_pos + l2 * np.array(
        [np.cos(theta1 + theta2), np.sin(theta1 + theta2)], dtype=np.float32
    )
    return base_pos, elbow_pos, ee_pos


def end_effector_position(theta1: float, theta2: float, l1: float, l2: float) -> np.ndarray:
    """Trả về vị trí end-effector, shape (2,)."""
    _, _, ee_pos = forward_kinematics(theta1, theta2, l1, l2)
    return ee_pos
