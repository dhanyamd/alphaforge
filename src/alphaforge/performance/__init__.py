"""(VI) PERFORMANCE ANALYSIS — separate skill from luck, close the loop.

After the fact, decompose what actually happened: how much came from the factor
bets you intended, how much from constraints, how much was noise. Two numbers tie
the whole machine together:

  * **Information Ratio** IR = active return / active risk — the report card.
  * **Fundamental law** IR ≈ IC · sqrt(breadth) — the two ways to get better:
    more skill per bet, or more *independent* bets.

And because factors decay, the `decay` module watches each factor's IC over time
and feeds the research loop: an edge that printed five years ago gets crowded out.
"""

from alphaforge.performance.metrics import performance_summary

__all__ = ["performance_summary"]
