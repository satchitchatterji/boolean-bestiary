# boolean-bestiary

`boolean-bestiary` provides a reusable PettingZoo Parallel environment for **n-player Boolean games**.

In a Boolean game, each agent controls one or more Boolean variables and has a private Boolean goal formula. All agents act simultaneously by setting their controlled variables, then each receives reward based on whether its goal is satisfied (minus optional action costs).

## Why Parallel API?

Boolean games are naturally simultaneous-action games. PettingZoo's Parallel API is designed for this setting, where all live agents provide actions at the same environment step.

## PettingZoo-Style API Docs

### Environment name

- Class: `BooleanGameEnv`
- Module: `boolean_bestiary.envs.boolean_game`
- API: PettingZoo `ParallelEnv`
- Metadata name: `boolean_bestiary_v0`

### Import

```python
from boolean_bestiary import BooleanGameEnv
```

### Arguments

- `num_agents: int`: Number of agents. Agents are named `agent_0 ... agent_{n-1}`.
- `variables: list[str] | None`: Global Boolean variable names. Defaults to `["x0", ..., "x{n-1}"]`.
- `controlled_variables: dict[str, list[str]] | None`: Agent-to-variable control map. Defaults to one variable per agent.
- `goals: dict[str, str] | None`: Agent-to-formula map. Defaults to each agent wanting its own variable true.
- `costs: dict[str, dict[str, float]] | None`: Agent variable costs when setting controlled variable to `True`. Defaults to `0.0`.
- `max_cycles: int`: Number of parallel steps before termination.
- `render_mode: str | None`: `None`, `"ansi"`, or `"human"`.
- `seed: int | None`: Optional initial RNG seed.

### Parallel API usage

```python
env = BooleanGameEnv(num_agents=3, max_cycles=2)
observations, infos = env.reset(seed=42)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

### Observation Space

Per-agent observation is a `gymnasium.spaces.Dict` with:

- `assignment`: `MultiBinary(len(variables))` global variable assignment
- `controlled`: `MultiBinary(len(variables))` mask of variables controlled by that agent
- `step`: `Discrete(max_cycles + 1)` current cycle index

### Action Space

Per-agent action space is `MultiBinary(k)`, where `k` is the number of variables controlled by that agent.
Each bit is the assigned truth value for one controlled variable at the current parallel step.

### Reward

For each agent:

- Base reward `1.0` if the agent's Boolean goal formula is satisfied, else `0.0`
- Subtract per-variable costs for controlled variables set to `True` by that agent in that step

### Starting State

- On `reset()`, all global variables are initialized to `False`.
- `agents` is reset to all `possible_agents`.

### Episode End

- `terminations[agent]` becomes `True` for all agents after `max_cycles` steps.
- `truncations[agent]` is `False` in the current implementation.

### Invalid Configurations

- Overlapping controlled variables are rejected at initialization.
- Invalid formulas are rejected (unsafe syntax, unknown variables, unsupported constants/operators).
- Invalid action shape/values are rejected during `step()`.

### Utility Methods

- `observe(agent)`: returns one agent's observation dict
- `state()`: returns global Boolean assignment as a `numpy.ndarray`
- `render()`: returns (or prints in `"human"` mode) a string like `step=1 assignment={x0=True, x1=False}`
- `close()`: no-op cleanup hook

## Installation

```bash
pip install -e .
```

For development/tests:

```bash
pip install -e '.[dev]'
```

## Quickstart

```python
from boolean_bestiary import BooleanGameEnv

env = BooleanGameEnv(num_agents=3, max_cycles=2)
obs, infos = env.reset(seed=42)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    print("actions:", actions)
    print("rewards:", rewards)

print("final assignment:", env.state())
print(env.render())  # step=2 assignment={x0=True, x1=False, x2=True}
env.close()
```

## Environment configuration

`BooleanGameEnv(...)` supports:

- `num_agents`: Number of agents, named `agent_0 ... agent_{n-1}`
- `variables`: Global Boolean variables (default: `x0, x1, ...`)
- `controlled_variables`: Mapping from agent to variables it can set
- `goals`: Mapping from agent to Boolean goal formula
- `costs`: Mapping from agent to per-variable activation costs
- `max_cycles`: Number of simultaneous-action rounds before termination
- `render_mode`: `\"human\"`, `\"ansi\"`, or `None`
- `seed`: Optional RNG seed

Defaults create a simple game where each agent controls one variable and tries to make it `True`.

### Example custom game

```python
from boolean_bestiary import BooleanGameEnv

env = BooleanGameEnv(
    num_agents=3,
    variables=["x0", "x1", "x2"],
    controlled_variables={
        "agent_0": ["x0"],
        "agent_1": ["x1"],
        "agent_2": ["x2"],
    },
    goals={
        "agent_0": "x0 and not x1",
        "agent_1": "x0 or x1",
        "agent_2": "x2 and (x0 | x1)",
    },
    costs={
        "agent_0": {"x0": 0.1},
        "agent_1": {"x1": 0.2},
        "agent_2": {"x2": 0.3},
    },
    max_cycles=1,
)
```

## Formula syntax

Goals accept safe Boolean formulas (no `eval`):

- Variables: `x0`, `foo`, `bar_1`
- Operators: `and`, `or`, `not`
- Parentheses: `(x0 and not x1) or x2`
- Constants: `True`, `False`
- Symbolic aliases: `&`, `|`, `~`, `!`

Unsupported syntax is rejected with clear errors.

## Observation and action spaces

- Action space: per-agent `MultiBinary(k)` where `k = len(controlled_variables[agent])`
- Observation space: per-agent `Dict` containing:
- `assignment`: `MultiBinary(len(variables))` global assignment
- `controlled`: `MultiBinary(len(variables))` ownership mask
- `step`: current cycle counter

The environment follows PettingZoo Parallel API and returns:
`observations, rewards, terminations, truncations, infos` from `step(actions)`.

## Running tests

```bash
pytest
```

The suite includes behavior tests and PettingZoo's `parallel_api_test`.

## Example script

```bash
python examples/random_policy.py
```

## Tabular IQL baseline

A minimal Independent Q-Learning baseline is included in:

- `examples/algorithms/iql.py`: reusable tabular IQL components
- `examples/train_iql.py`: runnable training script on `BooleanGameEnv`
- `examples/assets/iql_training_flow.mmd`: training loop diagram (Mermaid)

Run it with:

```bash
python examples/train_iql.py
```

Training artifacts are written to `examples/assets/`:

- `iql_training_returns.csv`: episode return + 100-episode moving average
- `iql_training_curve.png`: plotted training curve (if `matplotlib` is installed)
