# boolean-bestiary
<p align="center"><img src="examples/assets/bb-logo.jpg" alt="boolean-bestiary logo" width="240" /></p>
`boolean-bestiary` provides a reusable PettingZoo Parallel environment for **n-player Boolean games**.

In a Boolean game, each agent controls one or more Boolean variables and has a private Boolean goal formula. All agents act simultaneously by setting their controlled variables, then each receives reward based on whether its goal is satisfied (minus optional action costs).

## Why Parallel API?

Boolean games are naturally simultaneous-action games. PettingZoo's Parallel API is designed for this setting, where all live agents provide actions at the same environment step.

## Installation

```bash
pip install -e .
```

For development/tests:

```bash
pip install -e '.[dev]'
```

For docs:

```bash
pip install -e '.[docs]'
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
print(env.render())
env.close()
```

## Documentation

Detailed PettingZoo-style API docs are in Sphinx under `docs/`.

Build HTML docs:

```bash
make -C docs html
```

Open: `docs/_build/html/index.html`

## Formula syntax

Goals accept safe Boolean formulas (no `eval`):

- Variables: `x0`, `foo`, `bar_1`
- Operators: `and`, `or`, `not`
- Parentheses: `(x0 and not x1) or x2`
- Constants: `True`, `False`
- Symbolic aliases: `&`, `|`, `~`, `!`

Unsupported syntax is rejected with clear errors.

## Running tests

```bash
pytest
```

The suite includes behavior tests and PettingZoo's `parallel_api_test`.

## Examples

```bash
python examples/random_policy.py
python examples/train_iql.py
```

Training artifacts are written to `examples/assets/`:

- `iql_training_returns.csv`
- `iql_training_curve.png`
- `iql_training_flow.mmd`

## References

- Harrenstein, P., van der Hoek, W., Meyer, J.-J. C., and Witteveen, C. (2001). *Boolean Games*. TARK 2001. [PDF](https://www.tark.org/proceedings/tark_jul8_01/p287-harrenstein.pdf)
- Harrenstein, P., van der Hoek, W., Meyer, J.-J. C., and Witteveen, C. (2002). *On Modal Logic Interpretations of Games*. ECAI 2002. [Record](https://research-portal.uu.nl/en/publications/on-modal-logic-interpretations-of-games)
- Gutierrez, J., Harrenstein, P., and Wooldridge, M. (2015). *Iterated Boolean Games*. *Artificial Intelligence*. [Article](https://www.sciencedirect.com/science/article/pii/S0890540115000267)
