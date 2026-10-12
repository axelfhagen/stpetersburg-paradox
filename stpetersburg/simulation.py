"""Simulating the St. Petersburg game.

Drawing every coin flip for 10,000 players x 10^9 games is out of reach, but
only the *sum* of m payoffs is ever needed. Among m games, the number that end
on flip 1 is Binomial(m, 1/2); of the rest, the number that end on flip 2 is
again Binomial(remaining, 1/2), and so on. An exact draw of the sum therefore
takes about log2(m) binomial draws, vectorised across all players.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def batch_sums(m: np.ndarray, rng: np.random.Generator, cap: float | None = None) -> np.ndarray:
    """Exact draw of the sum of m[i] i.i.d. payoffs, for each i.

    `cap` models a casino with a finite bankroll: no single payout exceeds it.
    """
    remaining = np.asarray(m, dtype=np.int64).copy()
    total = np.zeros(remaining.shape)
    k = 1
    while remaining.any():
        payoff = 2.0 ** k
        if cap is not None and payoff >= cap:
            total += remaining * cap  # every game still running pays the cap
            break
        ended = rng.binomial(remaining, 0.5)
        total += ended * payoff
        remaining -= ended
        k += 1
    return total


def running_means(n_paths: int, checkpoints: np.ndarray, rng: np.random.Generator,
                  cap: float | None = None) -> np.ndarray:
    """Average payoff per game for every player at every checkpoint.

    Returns an array of shape (len(checkpoints), n_paths).
    """
    increments = np.diff(np.concatenate([[0], checkpoints]))
    totals = np.zeros(n_paths)
    out = np.empty((len(checkpoints), n_paths))
    for i, m in enumerate(increments):
        totals += batch_sums(np.full(n_paths, m), rng, cap)
        out[i] = totals / checkpoints[i]
    return out


def make_checkpoints(max_exp: int, num: int = 360) -> np.ndarray:
    """Log-spaced game counts from 1 to 10^max_exp, always including each power of ten."""
    grid = np.round(np.logspace(0, max_exp, num)).astype(np.int64)
    decades = 10 ** np.arange(0, max_exp + 1, dtype=np.int64)
    return np.unique(np.concatenate([grid, decades]))


def capped_ev(bank: np.ndarray | float) -> np.ndarray:
    """Expected payoff when the casino can pay at most `bank` dollars."""
    bank = np.asarray(bank, dtype=float)
    K = np.floor(np.log2(bank))
    return K + bank / 2.0 ** K


def certainty_equivalent(wealth: float, utility: str) -> float:
    """Most a player with this wealth and utility ("log" or "sqrt") would pay to play once."""
    ks = np.arange(1, 400)
    p = 0.5 ** ks
    prizes = 2.0 ** ks
    u = np.log if utility == "log" else np.sqrt
    base = u(wealth)

    def gap(price: float) -> float:
        with np.errstate(divide="ignore"):
            return float(np.sum(p * u(wealth - price + prizes)) - base)

    hi = wealth + 2 - 1e-9  # beyond this the player could end up with negative wealth
    return hi if gap(hi) > 0 else brentq(gap, 0.0, hi)


def pick_journey(rng: np.random.Generator, games: int, candidates: int = 400):
    """Pick one player whose history illustrates the game well.

    Prefers a single dominant jackpot landing around the middle of the run, so
    the before and after are both visible. Returns (flips, payoffs).
    """
    k = rng.geometric(0.5, size=(candidates, games))
    pay = 2.0 ** k
    top = pay.max(axis=1)
    when = pay.argmax(axis=1) / games
    big = (pay >= top[:, None] / 8).sum(axis=1)
    ok = (when > 0.35) & (when < 0.7) & (top >= 2 ** 11) & (top <= 2 ** 14)
    score = np.where(ok, big + top / 2 ** 14, 0)
    i = int(np.argmax(score))
    return k[i], pay[i]
