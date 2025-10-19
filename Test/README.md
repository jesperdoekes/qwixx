# Qwixx (Python)

A working Python port of the core Qwixx client logic (rules, phases, closures, scoring).
Includes a small CLI so you can play locally with 2+ players.

## Run
    python -m qwixx.cli
    # or
    python -m qwixx.cli Alice Bob

## Highlights
- Phases: initial → whiteSum → colorDice → complete
- White-sum (all players), Color-dice (active player)
- Sequence rules: red/yellow ascend, green/blue descend
- Row closures: deferred in white phase, immediate in color phase
- End when 2 rows closed or a player has 4 penalties
- Scoring: n(n+1)/2 per row, +1 count lock bonus only for the player who closed the row, −5 per penalty
