Examples
========

Random Policy Example
---------------------

Run:

.. code-block:: bash

   python examples/random_policy.py

Tabular IQL Baseline
--------------------

Included files:

- ``examples/algorithms/iql.py``: reusable tabular IQL components
- ``examples/train_iql.py``: runnable training script on ``BooleanGameEnv``
- ``examples/assets/iql_training_flow.mmd``: Mermaid training-loop diagram

Run training:

.. code-block:: bash

   python examples/train_iql.py

Training Artifacts
------------------

Written to ``examples/assets/``:

- ``iql_training_returns.csv``: episode return + 100-episode moving average
- ``iql_training_curve.png``: plotted training curve (if ``matplotlib`` is installed)
