Formula Syntax
==============

Goal formulas are parsed with a safe AST-based evaluator (no raw ``eval``).

Supported Syntax
----------------

- Variables: ``x0``, ``foo``, ``bar_1``
- Operators: ``and``, ``or``, ``not``
- Parentheses: ``(x0 and not x1) or x2``
- Constants: ``True``, ``False``
- Symbolic aliases: ``&`` (and), ``|`` (or), ``~`` and ``!`` (not)

Unsupported syntax is rejected with a clear error message.

Example
-------

.. code-block:: python

   goals = {
       "agent_0": "x0 and not x1",
       "agent_1": "x0 or x1",
       "agent_2": "x2 and (x0 | x1)",
   }
