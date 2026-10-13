"""Static figures 01-11, grouped by theme."""

from .convergence import plot_fan, plot_heatmap, plot_ridgeline
from .outcomes import plot_gamblers, plot_profit_odds, plot_who_wins, plot_worth
from .player import plot_journey
from .theory import plot_bankroll, plot_tail, plot_utility

__all__ = [
    "plot_bankroll", "plot_fan", "plot_gamblers", "plot_heatmap", "plot_journey",
    "plot_profit_odds", "plot_ridgeline", "plot_tail", "plot_utility", "plot_who_wins",
    "plot_worth",
]
