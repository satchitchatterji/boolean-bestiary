from __future__ import annotations

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from boolean_bestiary import BooleanGameEnv


def test_construct_multiple_agent_counts() -> None:
    for n in [2, 3, 5]:
        env = BooleanGameEnv(num_agents=n)
        assert len(env.possible_agents) == n
        assert env.agents == env.possible_agents


def test_reset_returns_valid_observations_and_infos() -> None:
    env = BooleanGameEnv(num_agents=3)
    observations, infos = env.reset(seed=123)

    assert set(observations.keys()) == set(env.possible_agents)
    assert set(infos.keys()) == set(env.possible_agents)

    for agent, obs in observations.items():
        assert env.observation_space(agent).contains(obs)
        assert infos[agent] == {}


def test_step_updates_assignment_from_simultaneous_actions() -> None:
    env = BooleanGameEnv(num_agents=3)
    env.reset()

    actions = {
        "agent_0": np.array([1], dtype=np.int8),
        "agent_1": np.array([0], dtype=np.int8),
        "agent_2": np.array([1], dtype=np.int8),
    }

    env.step(actions)
    np.testing.assert_array_equal(env.state(), np.array([1, 0, 1], dtype=np.int8))


def test_rewards_match_goal_satisfaction() -> None:
    goals = {
        "agent_0": "x0 and not x1",
        "agent_1": "x0 or x1",
    }
    env = BooleanGameEnv(num_agents=2, goals=goals)
    env.reset()

    _, rewards, _, _, _ = env.step(
        {
            "agent_0": np.array([1], dtype=np.int8),
            "agent_1": np.array([0], dtype=np.int8),
        }
    )

    assert rewards["agent_0"] == 1.0
    assert rewards["agent_1"] == 1.0


def test_costs_subtracted_correctly() -> None:
    costs = {
        "agent_0": {"x0": 0.25},
        "agent_1": {"x1": 0.5},
    }
    env = BooleanGameEnv(num_agents=2, costs=costs)
    env.reset()

    _, rewards, _, _, _ = env.step(
        {
            "agent_0": np.array([1], dtype=np.int8),
            "agent_1": np.array([1], dtype=np.int8),
        }
    )

    assert rewards["agent_0"] == pytest.approx(0.75)
    assert rewards["agent_1"] == pytest.approx(0.5)


def test_invalid_formula_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported syntax|Invalid formula syntax|Unknown variable"):
        BooleanGameEnv(
            num_agents=2,
            goals={
                "agent_0": "x0 and __import__('os').system('echo bad')",
                "agent_1": "x1",
            },
        )


def test_overlapping_controlled_variables_rejected() -> None:
    with pytest.raises(ValueError, match="Overlapping control"):
        BooleanGameEnv(
            num_agents=2,
            controlled_variables={
                "agent_0": ["x0"],
                "agent_1": ["x0"],
            },
        )


def test_parallel_api() -> None:
    env = BooleanGameEnv(num_agents=3, max_cycles=3)
    parallel_api_test(env, num_cycles=20)
