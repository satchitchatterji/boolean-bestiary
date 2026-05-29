"""Run one random episode in a 3-agent Boolean game."""

from __future__ import annotations

from boolean_bestiary import BooleanGameEnv


def main() -> None:
    env = BooleanGameEnv(num_agents=3, max_cycles=1, seed=7)
    observations, infos = env.reset(seed=7)
    print("Initial observations:", observations)
    print("Infos:", infos)

    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        print("Actions:", actions)
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print("Observations:", observations)
        print("Rewards:", rewards)
        print("Terminations:", terminations)
        print("Truncations:", truncations)
        print("Infos:", infos)

    print("Final state:", env.state())
    print("Render:", env.render())
    env.close()


if __name__ == "__main__":
    main()
