# Alpha-Beta Go Agent
A standalone implementation of an Alpha-Beta search agent for the game of Go (simplified 5x5 board version).

This was originally developed for coursework but is shared here as a clean, self-contained version of the **agent only**.

---

## Repo Layout

src/  
  myplayer3.py   # Alpha-Beta agent implementation

No course harness files (e.g., read.py, write.py, host.py) are included here.

---

## Usage

This repository contains only the standalone Alpha-Beta agent logic.  
It depends on the course harness (read.py, write.py, host.py) at runtime,  
but those files are not included in this repo.

### Running with the harness

Clone/download your course framework locally, then run:

    python src/myplayer3.py

The AlphaBetaPlayer class will interact with the provided GO environment via readInput / writeOutput.

---

## Notes
- Do not commit any external framework files here.  
- Only src/myplayer3.py is original work authored for this repo.  

---

## License
MIT License. See LICENSE for details.
