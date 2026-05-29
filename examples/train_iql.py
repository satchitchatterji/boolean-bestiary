"""Train a tabular IQL baseline on BooleanGameEnv."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from boolean_bestiary import BooleanGameEnv
from examples.algorithms.iql import (
    IQLConfig,
    TabularIQLAgent,
    binary_vector_to_index,
    index_to_binary_vector,
)


def train_iql(
    episodes: int = 2000,
    max_cycles: int = 1,
    seed: int = 0,
) -> None:
    assets_dir = Path(__file__).resolve().parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    env = BooleanGameEnv(num_agents=3, max_cycles=max_cycles, seed=seed)
    config = IQLConfig(
        learning_rate=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=max(episodes // 2, 1),
    )

    n_state_bits = len(env.variables)
    n_states = 1 << n_state_bits

    agents: dict[str, TabularIQLAgent] = {}
    for i, agent in enumerate(env.possible_agents):
        k = len(env.controlled_variables[agent])
        n_actions = 1 << k
        agents[agent] = TabularIQLAgent(
            n_states=n_states,
            n_actions=n_actions,
            config=config,
            seed=seed + i + 1,
        )

    reward_history: list[float] = []
    global_step = 0

    for episode in range(episodes):
        observations, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0

        while not done:
            states = {
                agent: binary_vector_to_index(observations[agent]["assignment"])
                for agent in env.agents
            }

            action_ids: dict[str, int] = {}
            joint_actions: dict[str, np.ndarray] = {}
            for agent in env.agents:
                act_id = agents[agent].select_action(states[agent], global_step)
                action_ids[agent] = act_id
                action_vec = index_to_binary_vector(
                    act_id, len(env.controlled_variables[agent])
                )
                joint_actions[agent] = action_vec

            next_obs, rewards, terminations, truncations, _ = env.step(joint_actions)

            done = all(terminations.values()) or all(truncations.values())
            next_states = {
                agent: binary_vector_to_index(next_obs[agent]["assignment"])
                for agent in next_obs
            }

            for agent in rewards:
                agents[agent].update(
                    state=states[agent],
                    action=action_ids[agent],
                    reward=float(rewards[agent]),
                    next_state=next_states[agent],
                    done=done,
                )
                episode_return += float(rewards[agent])

            observations = next_obs
            global_step += 1

        reward_history.append(episode_return)

        if (episode + 1) % max(episodes // 10, 1) == 0:
            window = reward_history[-100:]
            mean_return = float(np.mean(window))
            print(
                f"Episode {episode + 1:4d}/{episodes} | "
                f"avg total return (last {len(window)}): {mean_return:.3f}"
            )

    print("\nTraining finished.")
    print("Config:", asdict(config))
    print("Final 100-episode mean total return:", float(np.mean(reward_history[-100:])))
    _save_training_curves(reward_history, assets_dir)

    # Greedy evaluation rollout
    eval_obs, _ = env.reset(seed=seed + 10_000)
    print("\nGreedy policy rollout:")
    while env.agents:
        eval_actions: dict[str, np.ndarray] = {}
        for agent in env.agents:
            s = binary_vector_to_index(eval_obs[agent]["assignment"])
            greedy_id = int(np.argmax(agents[agent].q_table[s]))
            eval_actions[agent] = index_to_binary_vector(
                greedy_id, len(env.controlled_variables[agent])
            )
        eval_obs, rewards, terms, truncs, _ = env.step(eval_actions)
        print("actions:", eval_actions, "rewards:", rewards, "done:", terms, truncs)

    print("Final state:", env.state())
    print("Render:", env.render())
    env.close()


def _save_training_curves(reward_history: list[float], assets_dir: Path) -> None:
    episodes = np.arange(1, len(reward_history) + 1, dtype=np.int32)
    returns = np.asarray(reward_history, dtype=np.float64)
    window = 100
    moving_avg = np.empty_like(returns)
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        moving_avg[i] = np.mean(returns[start : i + 1])

    csv_path = assets_dir / "iql_training_returns.csv"
    np.savetxt(
        csv_path,
        np.column_stack([episodes, returns, moving_avg]),
        delimiter=",",
        header="episode,total_return,moving_avg_100",
        comments="",
    )
    print(f"Saved training returns CSV: {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipped PNG curve export.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(episodes, returns, alpha=0.35, label="episode return")
    ax.plot(episodes, moving_avg, linewidth=2.0, label="moving avg (100)")
    ax.set_title("Tabular IQL Training Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total return (sum over agents)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    png_path = assets_dir / "iql_training_curve.png"
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Saved training curve PNG: {png_path}")


if __name__ == "__main__":
    train_iql()
