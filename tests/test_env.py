"""Test RobotArm2DEnv: API Gymnasium, shape, reward, random rollout."""

import numpy as np
import pytest

from robot_arm.env import RobotArm2DEnv
from robot_arm.kinematics import end_effector_position


@pytest.fixture
def env() -> RobotArm2DEnv:
    return RobotArm2DEnv(max_steps=50)


def test_spaces_shape_and_dtype(env: RobotArm2DEnv) -> None:
    assert env.observation_space.shape == (10,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (2,)
    assert env.action_space.dtype == np.float32
    assert np.allclose(env.action_space.low, -1.0)
    assert np.allclose(env.action_space.high, 1.0)


def test_reset_returns_obs_and_info(env: RobotArm2DEnv) -> None:
    obs, info = env.reset(seed=42)
    assert obs.shape == (10,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert "distance" in info
    assert "end_effector" in info
    assert "target" in info


def test_reset_seed_reproducible() -> None:
    env_a = RobotArm2DEnv()
    env_b = RobotArm2DEnv()
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    np.testing.assert_allclose(obs_a, obs_b)


def test_step_returns_5_tuple(env: RobotArm2DEnv) -> None:
    env.reset(seed=0)
    action = env.action_space.sample()
    result = env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (10,)
    assert obs.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_reward_is_negative_distance(env: RobotArm2DEnv) -> None:
    env.reset(seed=1)
    action = env.action_space.sample()
    _, reward, _, _, info = env.step(action)
    assert reward == pytest.approx(-info["distance"])

    ee = end_effector_position(env.theta[0], env.theta[1], env.l1, env.l2)
    expected_dist = float(np.linalg.norm(env.target - ee))
    assert info["distance"] == pytest.approx(expected_dist)
    assert reward == pytest.approx(-expected_dist)


def test_terminated_always_false(env: RobotArm2DEnv) -> None:
    env.reset(seed=2)
    for _ in range(50):
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert terminated is False


def test_truncated_at_max_steps() -> None:
    max_steps = 20
    env = RobotArm2DEnv(max_steps=max_steps)
    env.reset(seed=3)
    truncated = False
    steps_taken = 0
    for _ in range(max_steps):
        _, _, _, truncated, _ = env.step(env.action_space.sample())
        steps_taken += 1
        if truncated:
            break
    assert truncated is True
    assert steps_taken == max_steps


def test_random_rollout_stays_in_observation_space(env: RobotArm2DEnv) -> None:
    obs, _ = env.reset(seed=7)
    assert env.observation_space.contains(obs)
    for _ in range(env.max_steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert np.isfinite(obs).all()
        if terminated or truncated:
            break


def test_target_within_workspace(env: RobotArm2DEnv) -> None:
    for seed in range(10):
        env.reset(seed=seed)
        radius = float(np.linalg.norm(env.target))
        assert env.l1 + env.l2 >= radius >= abs(env.l1 - env.l2)


@pytest.fixture
def circle_env() -> RobotArm2DEnv:
    return RobotArm2DEnv(max_steps=50, trajectory="circle")


def test_invalid_trajectory_raises() -> None:
    with pytest.raises(ValueError):
        RobotArm2DEnv(trajectory="zigzag")


def test_circle_target_moves_each_step(circle_env: RobotArm2DEnv) -> None:
    circle_env.reset(seed=4)
    prev_target = circle_env.target.copy()
    for _ in range(5):
        circle_env.step(circle_env.action_space.sample())
        assert not np.allclose(circle_env.target, prev_target)
        prev_target = circle_env.target.copy()


def test_circle_target_stays_on_circle(circle_env: RobotArm2DEnv) -> None:
    circle_env.reset(seed=5)
    center = circle_env._circle.center
    radius = circle_env._circle.radius
    for _ in range(50):
        circle_env.step(circle_env.action_space.sample())
        dist_to_center = float(np.linalg.norm(circle_env.target - center))
        assert dist_to_center == pytest.approx(radius, abs=1e-5)


def test_circle_obs_contains_target_and_error(circle_env: RobotArm2DEnv) -> None:
    circle_env.reset(seed=6)
    obs, _, _, _, info = circle_env.step(circle_env.action_space.sample())
    np.testing.assert_allclose(obs[6:8], circle_env.target, rtol=1e-5)
    ee = end_effector_position(
        circle_env.theta[0], circle_env.theta[1], circle_env.l1, circle_env.l2
    )
    np.testing.assert_allclose(obs[8:10], circle_env.target - ee, rtol=1e-4, atol=1e-6)


def test_circle_random_rollout_in_observation_space(
    circle_env: RobotArm2DEnv,
) -> None:
    obs, _ = circle_env.reset(seed=8)
    assert circle_env.observation_space.contains(obs)
    for _ in range(circle_env.max_steps):
        obs, reward, terminated, truncated, _ = circle_env.step(
            circle_env.action_space.sample()
        )
        assert circle_env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
