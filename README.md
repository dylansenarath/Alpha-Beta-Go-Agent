\# Alpha-Beta Go Agent



Minimax with alpha-beta pruning agent for the game of Go (or a simplified version). Includes baseline opponent strategies, evaluation suite, and metrics.



\## Features

\- Alpha-beta minimax search with optional heuristics

\- Opponents: random, greedy, aggressive

\- Depth-limited search, with optional iterative deepening



\## Quickstart

```bash

python -m venv .venv \&\& .\\.venv\\Scripts\\activate  # Windows

pip install -r requirements.txt



\# Example match

python src/run\_match.py --agent alphabeta --opponent greedy --depth 3



\# Batch evaluation

python src/eval\_suite.py --games 20 --depth 4



