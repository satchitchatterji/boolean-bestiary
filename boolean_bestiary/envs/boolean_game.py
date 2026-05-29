"""PettingZoo ParallelEnv for n-player Boolean games."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class FormulaError(ValueError):
    """Raised when a Boolean formula is invalid."""


@dataclass(frozen=True)
class ParsedFormula:
    """Pre-validated AST wrapper for a Boolean formula."""

    source: str
    tree: ast.Expression


class BooleanFormulaEvaluator:
    """Safe Boolean formula parser and evaluator."""

    _ALLOWED_NODES = (
        ast.Expression,
        ast.BoolOp,
        ast.UnaryOp,
        ast.Name,
        ast.Constant,
        ast.Load,
        ast.And,
        ast.Or,
        ast.Not,
    )

    @staticmethod
    def _normalize(expr: str) -> str:
        normalized = expr.replace("&", " and ").replace("|", " or ").replace("~", " not ")
        normalized = normalized.replace("!", " not ")
        return normalized

    @classmethod
    def parse(cls, expr: str, allowed_variables: set[str]) -> ParsedFormula:
        if not isinstance(expr, str) or not expr.strip():
            raise FormulaError("Formula must be a non-empty string.")

        normalized = cls._normalize(expr)
        try:
            tree = ast.parse(normalized, mode="eval")
        except SyntaxError as exc:
            raise FormulaError(f"Invalid formula syntax: {expr!r}. {exc.msg}") from exc

        for node in ast.walk(tree):
            if not isinstance(node, cls._ALLOWED_NODES):
                raise FormulaError(
                    f"Unsupported syntax in formula {expr!r}: {type(node).__name__}. "
                    "Only and/or/not, parentheses, variables, and True/False are allowed."
                )
            if isinstance(node, ast.Name) and node.id not in allowed_variables:
                raise FormulaError(
                    f"Unknown variable {node.id!r} in formula {expr!r}. "
                    f"Allowed variables: {sorted(allowed_variables)}"
                )
            if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
                raise FormulaError(
                    f"Unsupported constant {node.value!r} in formula {expr!r}. "
                    "Only True and False are allowed."
                )

        return ParsedFormula(source=expr, tree=tree)

    @classmethod
    def evaluate(cls, parsed: ParsedFormula, assignment: dict[str, bool]) -> bool:
        def _eval(node: ast.AST) -> bool:
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Name):
                return bool(assignment[node.id])
            if isinstance(node, ast.Constant):
                return bool(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return not _eval(node.operand)
            if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                return all(_eval(value) for value in node.values)
            if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
                return any(_eval(value) for value in node.values)
            raise FormulaError(f"Unsupported AST node during evaluation: {type(node).__name__}")

        return _eval(parsed.tree)


class BooleanGameEnv(ParallelEnv[str, dict[str, np.ndarray], np.ndarray]):
    """Parallel PettingZoo environment for n-player Boolean games."""

    metadata = {
        "name": "boolean_bestiary_v0",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        num_agents: int,
        variables: list[str] | None = None,
        controlled_variables: dict[str, list[str]] | None = None,
        goals: dict[str, str] | None = None,
        costs: dict[str, dict[str, float]] | None = None,
        max_cycles: int = 1,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        if num_agents <= 0:
            raise ValueError("num_agents must be > 0.")
        if max_cycles <= 0:
            raise ValueError("max_cycles must be > 0.")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}.")

        self.n_agents = num_agents
        self.render_mode = render_mode
        self.max_cycles = max_cycles
        self._seed = seed

        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]

        if variables is None:
            self.variables = [f"x{i}" for i in range(num_agents)]
        else:
            self.variables = list(variables)
        self._validate_variables(self.variables)

        self._var_index = {name: i for i, name in enumerate(self.variables)}

        if controlled_variables is None:
            controlled_variables = {
                f"agent_{i}": [f"x{i}"]
                for i in range(num_agents)
            }
        self.controlled_variables = self._validate_controlled_variables(controlled_variables)

        if goals is None:
            goals = {
                f"agent_{i}": f"x{i}"
                for i in range(num_agents)
            }
        self.goals = self._validate_goals(goals)

        self.costs = self._validate_costs(costs)
        self._parsed_goals = {
            agent: BooleanFormulaEvaluator.parse(goal, set(self.variables))
            for agent, goal in self.goals.items()
        }

        self._rng = np.random.default_rng(seed)
        self._cycle = 0
        self._assignment = np.zeros(len(self.variables), dtype=np.int8)

        self.action_spaces = {
            agent: spaces.MultiBinary(len(self.controlled_variables[agent]))
            for agent in self.possible_agents
        }

        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "assignment": spaces.MultiBinary(len(self.variables)),
                    "controlled": spaces.MultiBinary(len(self.variables)),
                    "step": spaces.Discrete(self.max_cycles + 1),
                }
            )
            for agent in self.possible_agents
        }

        self._controlled_masks = {
            agent: self._build_controlled_mask(agent)
            for agent in self.possible_agents
        }

    def _validate_variables(self, variables: list[str]) -> None:
        if not variables:
            raise ValueError("variables must contain at least one variable.")
        if len(set(variables)) != len(variables):
            raise ValueError("variables must be unique.")
        invalid = [name for name in variables if not _VALID_NAME_RE.match(name)]
        if invalid:
            raise ValueError(
                f"Invalid variable names: {invalid}. Names must match pattern {_VALID_NAME_RE.pattern}."
            )

    def _validate_controlled_variables(
        self, controlled_variables: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        if set(controlled_variables.keys()) != set(self.possible_agents):
            raise ValueError(
                "controlled_variables keys must match agents exactly: "
                f"{self.possible_agents}"
            )

        seen_controllers: dict[str, str] = {}
        validated: dict[str, list[str]] = {}

        for agent in self.possible_agents:
            var_list = list(controlled_variables[agent])
            if len(var_list) == 0:
                raise ValueError(f"{agent} must control at least one variable.")
            if len(set(var_list)) != len(var_list):
                raise ValueError(f"{agent} has duplicate controlled variables: {var_list}")

            for var in var_list:
                if var not in self._var_index:
                    raise ValueError(f"{agent} controls unknown variable {var!r}.")
                if var in seen_controllers:
                    other = seen_controllers[var]
                    raise ValueError(
                        f"Overlapping control is not supported: variable {var!r} is controlled by "
                        f"both {other!r} and {agent!r}."
                    )
                seen_controllers[var] = agent

            validated[agent] = var_list

        return validated

    def _validate_goals(self, goals: dict[str, str]) -> dict[str, str]:
        if set(goals.keys()) != set(self.possible_agents):
            raise ValueError(f"goals keys must match agents exactly: {self.possible_agents}")
        return {agent: str(goals[agent]) for agent in self.possible_agents}

    def _validate_costs(
        self, costs: dict[str, dict[str, float]] | None
    ) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        raw = costs or {}

        for agent in self.possible_agents:
            per_agent = dict(raw.get(agent, {}))
            validated: dict[str, float] = {}
            for var, value in per_agent.items():
                if var not in self.controlled_variables[agent]:
                    raise ValueError(
                        f"Cost variable {var!r} for {agent} must be one of controlled variables "
                        f"{self.controlled_variables[agent]}"
                    )
                validated[var] = float(value)

            for var in self.controlled_variables[agent]:
                validated.setdefault(var, 0.0)

            result[agent] = validated

        return result

    def _build_controlled_mask(self, agent: str) -> np.ndarray:
        mask = np.zeros(len(self.variables), dtype=np.int8)
        for var in self.controlled_variables[agent]:
            mask[self._var_index[var]] = 1
        return mask

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, dict[str, np.ndarray | int]], dict[str, dict[str, Any]]]:
        del options
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self._cycle = 0
        self._assignment = np.zeros(len(self.variables), dtype=np.int8)

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def observe(self, agent: str) -> dict[str, np.ndarray | int]:
        if agent not in self.possible_agents:
            raise ValueError(f"Unknown agent {agent!r}.")
        return {
            "assignment": self._assignment.copy(),
            "controlled": self._controlled_masks[agent].copy(),
            "step": int(self._cycle),
        }

    def state(self) -> np.ndarray:
        return self._assignment.copy()

    def step(
        self, actions: dict[str, np.ndarray | list[int] | int]
    ) -> tuple[
        dict[str, dict[str, np.ndarray | int]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        missing = set(self.agents) - set(actions.keys())
        if missing:
            raise ValueError(f"Missing actions for agents: {sorted(missing)}")

        for agent in self.agents:
            self._apply_action(agent, actions[agent])

        assignment_dict = {
            var: bool(self._assignment[idx])
            for var, idx in self._var_index.items()
        }

        rewards: dict[str, float] = {}
        for agent in self.agents:
            goal_value = BooleanFormulaEvaluator.evaluate(self._parsed_goals[agent], assignment_dict)
            reward = 1.0 if goal_value else 0.0

            agent_action = np.asarray(actions[agent], dtype=np.int8).reshape(-1)
            for i, var in enumerate(self.controlled_variables[agent]):
                if int(agent_action[i]) == 1:
                    reward -= self.costs[agent][var]
            rewards[agent] = reward

        self._cycle += 1
        terminated = self._cycle >= self.max_cycles

        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        observations = {agent: self.observe(agent) for agent in self.agents}

        if terminated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _apply_action(self, agent: str, action: np.ndarray | list[int] | int) -> None:
        action_arr = np.asarray(action, dtype=np.int8).reshape(-1)
        expected_size = len(self.controlled_variables[agent])

        if action_arr.size != expected_size:
            raise ValueError(
                f"Invalid action size for {agent}: expected {expected_size}, got {action_arr.size}."
            )
        if np.any((action_arr != 0) & (action_arr != 1)):
            raise ValueError(f"Action for {agent} must contain only 0/1 values. Got {action_arr}.")

        for i, var in enumerate(self.controlled_variables[agent]):
            self._assignment[self._var_index[var]] = int(action_arr[i])

    def render(self) -> str | None:
        assignment_str = ", ".join(
            f"{var}={bool(self._assignment[self._var_index[var]])}" for var in self.variables
        )
        text = f"step={self._cycle} assignment={{{assignment_str}}}"

        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        return None
