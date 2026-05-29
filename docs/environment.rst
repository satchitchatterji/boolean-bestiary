Environment
===========

Environment Name
----------------

- Class: ``BooleanGameEnv``
- Module: ``boolean_bestiary.envs.boolean_game``
- API: PettingZoo ``ParallelEnv``
- Metadata name: ``boolean_bestiary_v0``

Import
------

.. code-block:: python

   from boolean_bestiary import BooleanGameEnv

Arguments
---------

- ``num_agents: int``: Number of agents. Agents are named ``agent_0 ... agent_{n-1}``.
- ``variables: list[str] | None``: Global Boolean variable names. Defaults to ``["x0", ..., "x{n-1}"]``.
- ``controlled_variables: dict[str, list[str]] | None``: Agent-to-variable control map. Defaults to one variable per agent.
- ``goals: dict[str, str] | None``: Agent-to-formula map. Defaults to each agent wanting its own variable true.
- ``costs: dict[str, dict[str, float]] | None``: Agent variable costs when setting controlled variable to ``True``. Defaults to ``0.0``.
- ``max_cycles: int``: Number of parallel steps before termination.
- ``render_mode: str | None``: ``None``, ``"ansi"``, or ``"human"``.
- ``seed: int | None``: Optional initial RNG seed.

Observation Space
-----------------

Per-agent observation is a ``gymnasium.spaces.Dict`` with:

- ``assignment``: ``MultiBinary(len(variables))`` global variable assignment
- ``controlled``: ``MultiBinary(len(variables))`` mask of variables controlled by that agent
- ``step``: ``Discrete(max_cycles + 1)`` current cycle index

Action Space
------------

Per-agent action space is ``MultiBinary(k)``, where ``k`` is the number of variables controlled by that agent.
Each bit is the assigned truth value for one controlled variable at the current parallel step.

Reward
------

For each agent:

- Base reward ``1.0`` if the agent goal formula is satisfied, else ``0.0``
- Subtract per-variable costs for controlled variables set to ``True`` by that agent in that step

Starting State
--------------

- On ``reset()``, all global variables are initialized to ``False``.
- ``agents`` is reset to all ``possible_agents``.

Episode End
-----------

- ``terminations[agent]`` becomes ``True`` for all agents after ``max_cycles`` steps.
- ``truncations[agent]`` is ``False`` in the current implementation.

Invalid Configurations
----------------------

- Overlapping controlled variables are rejected at initialization.
- Invalid formulas are rejected (unsafe syntax, unknown variables, unsupported constants/operators).
- Invalid action shape/values are rejected during ``step()``.

Utility Methods
---------------

- ``observe(agent)``: returns one agent observation dict
- ``state()``: returns global Boolean assignment as a ``numpy.ndarray``
- ``render()``: returns (or prints in ``"human"`` mode) a string like ``step=1 assignment={x0=True, x1=False}``
- ``close()``: no-op cleanup hook

Parallel API Loop
-----------------

.. code-block:: python

   env = BooleanGameEnv(num_agents=3, max_cycles=2)
   observations, infos = env.reset(seed=42)

   while env.agents:
       actions = {agent: env.action_space(agent).sample() for agent in env.agents}
       observations, rewards, terminations, truncations, infos = env.step(actions)

API Reference
-------------

.. autoclass:: boolean_bestiary.envs.boolean_game.BooleanGameEnv
   :members:
   :undoc-members:
