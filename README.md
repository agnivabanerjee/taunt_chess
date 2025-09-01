# Taunting Stockfish – Cinematic Chess Prophecies

A head-to-head between a strong Stockfish and a weaker Stockfish where the strong side **taunts** with “prophecies” (predicted tactical/positional events) and **steers** toward making them come true. Includes a fast **shared look-ahead tree** (reuse mode) with **motif-aware widening**, or a classic “per-event” search (legacy mode). Progress bars and an event-odds panel are available for debugging.

## Requirements

* Python 3.9+
* [python-chess](https://pypi.org/project/python-chess/)
* A **Stockfish** binary available on your system `PATH`
  (or set `STOCKFISH_PATH=/path/to/stockfish` in your environment)

## Install

### Using uv (recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository and setup
git clone https://github.com/agniva/taunt_chess
cd taunt_chess

# Install dependencies with uv (creates virtual environment automatically)
uv sync

# Activate the environment (if needed)
# uv shell

# Run the project
uv run python match.py --unicode-board
```

### Alternative: Using pip

```bash
# in a fresh virtualenv is recommended
pip install python-chess

# optional: pin
# pip install "python-chess>=1.999"
```

### System Requirements

Ensure Stockfish is installed and visible:
* **macOS (brew)**: `brew install stockfish`
* **Ubuntu**: `sudo apt-get install stockfish`  
* **Windows**: Download from [Stockfish releases](https://stockfishchess.org/download/) and add to PATH
* **Alternative**: Set `STOCKFISH_PATH=/path/to/stockfish` environment variable

## Project layout (key files)

```
engine_utils.py      # Stockfish helpers (make_engine, play_best, policy_moves, softmax)
events.py            # Event definitions (original + EASY_EVENTS) and probability fn
policy_tree.py       # Shared look-ahead tree with motif-aware widening (reuse mode)
progress_utils.py    # Lightweight progress bars/markers (debug only)
taunter.py           # Core taunting/steering logic (this repo’s “brain”)
match.py             # CLI entry point to run a game and watch prophecies unfold
```

## Quickstart

Run with defaults (strong plays White, weak plays Black):

```bash
uv run python match.py --unicode-board
```

See progress bars & markers:

```bash
uv run python match.py --unicode-board --debug
```

Use **higher-frequency test events** (great for demos):

```bash
uv run python match.py --easy-events --unicode-board --debug
```

Show the **per-event probability** panel each ply:

```bash
uv run python match.py --show-event-probs --unicode-board
```

## Modes

### Reuse / Shared-tree (fast, recommended)

Build one shared look-ahead tree per scan; compute all event odds at once; reuse for steering.

```bash
uv run python match.py --reuse --unicode-board --show-event-probs --debug
```

Tuning (defaults shown):

```
--plies 6 --root-depth 12 --inner-depth 10
--root-topk 4 --inner-topk 3
--widen-checks 4 --widen-captures 4 --widen-hooks 2 --widen-lifts 2 --widen-cap 6
--quick-eval-depth 6
```

### Legacy (no reuse; per-event scans)

Restores the original behavior where each event is searched separately (slower, but useful for testing search heuristics). **Do not pass `--reuse`.**

```bash
uv run python match.py --unicode-board --debug
```

If you want steering to scan only on schedule (not every move), add:

```
--respect-scan-schedule-for-steering
```

## Event sets

* **Default (cinematic)**: rarer, more dramatic patterns (forks, ring breach, etc.)
* **Easy / test set** (fires often): checks, pawn captures, center entry, ring poke, pawn to 5th, rook on open file.

Select easy set:

```bash
uv run python match.py --easy-events
```

## Scheduling scans

Only scan/announce prophecies every **K** moves:

```bash
# every 10 full moves (move numbers 1, 11, 21, …)
uv run python match.py --scan-every-fullmoves 10

# every 30 plies (half-moves) – overrides fullmove schedule
uv run python match.py --scan-every-plies 30
```

To ensure steering doesn’t do extra scans between scheduled scans:

```
--respect-scan-schedule-for-steering
```

## Steering & variety

* Steering strength (bonus in centipawns per +1.0 event prob): `--lam 25.0`
* Don’t play a steered move if it drops eval by >N cp vs best: `--trust-drop 30`
* After a prophecy **fulfills/averts**, the same event is **temporarily banned** from retargeting to keep variety. Control the window:

```bash
uv run python match.py --avoid-retarget-plies 6
```

(Defaults to the prophecy horizon.)

## Display options

* Pretty board: `--unicode-board`
* Recompute & show **active prophecy** odds each ply: `--show-probs`
* Show **all event probabilities** each ply: `--show-event-probs`
* Control display scan budget:

  * `--disp-depth` / `--disp-topk` (for active prophecies)
  * `--event-disp-depth` / `--event-disp-topk` (for the panel)

## Performance tips

* Faster runs: lower depths and branching

  ```
  --strong-depth 12 --weak-depth 6 --plies 5 --root-topk 3 --inner-topk 2
  ```
* Use `--reuse` to avoid repeated engine calls; keep `--debug` off unless needed.
* In legacy mode, progress bars appear only if `--debug` is set.

## Troubleshooting

* **Stockfish not found**: add it to `PATH` or set `STOCKFISH_PATH`.
* **Unicode board issues**: drop `--unicode-board` and use ASCII.
* **Per-event progress bars missing (legacy mode)**: ensure `--debug` is set and you are **not** using `--reuse`.
* **“mutable default … use default\_factory” error**: The repo uses `field(default_factory=TreeParams)` in `ReuseCfg`. Make sure your `taunter.py` matches the latest version.

## Examples

Reuse + easy events + panel:

```bash
uv run python match.py --reuse --easy-events --show-event-probs \
  --strong-as white --strong-depth 16 --weak-depth 8 --weak-skill 2 \
  --unicode-board --debug
```

Legacy + frequent scans + progress bars:

```bash
uv run python match.py --easy-events --scan-every-fullmoves 1 \
  --strong-as white --strong-depth 16 --weak-depth 8 --weak-skill 2 \
  --unicode-board --debug
```

