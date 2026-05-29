# Example Assets

This folder stores static assets used by example scripts and docs.

- `iql_training_flow.mmd`: Mermaid flowchart for the tabular Independent Q-Learning training loop in `examples/train_iql.py`.
- `iql_training_returns.csv`: Per-episode training returns exported by `examples/train_iql.py` with columns:
  `episode,total_return,moving_avg_100`.
- `iql_training_curve.png`: Rendered training curve plot exported by `examples/train_iql.py` (requires `matplotlib`).

To render the Mermaid diagram, use any Mermaid-compatible viewer/editor.
