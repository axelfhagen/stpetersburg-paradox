"""Why the expectation is infinite, and the two classic ways to make it finite."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from ..formatting import count, dollars, money
from ..simulation import capped_ev, certainty_equivalent
from ..style import (BLUE, DASHED, INK_2, MUTED, ORANGE, SURFACE, header, log_money_axis,
                     save)


def plot_tail(outdir: Path, rng: np.random.Generator, note: str, draws: int = 20_000_000) -> None:
    """Figure 04: simulated vs. theoretical tail P(X >= x) = 2/x, and share of games vs. money by k."""
    k = rng.geometric(0.5, size=draws)
    kmax = int(k.max())
    ks = np.arange(1, kmax + 1)
    freq = np.bincount(k, minlength=kmax + 1)[1:]
    ccdf = freq[::-1].cumsum()[::-1] / draws
    payout = freq * 2.0 ** ks

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.76, bottom=0.13, wspace=0.22)

    xs = 2.0 ** ks
    a1.plot(xs, 2.0 / xs, color=INK_2, lw=1.4, ls=DASHED, label="Theory: P(X ≥ x) = 2/x")
    a1.scatter(xs, ccdf, s=44, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
               zorder=5, label=f"Simulated ({count(draws)} games)")
    a1.set_xscale("log", base=2)
    a1.set_yscale("log")
    a1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: money(float(f"{v:.2g}"))))
    a1.set_xticks([2 ** e for e in range(1, kmax + 1, 4)])
    a1.set_xlabel("Payoff x")
    a1.set_ylabel("Chance of winning at least x")
    a1.set_title("A power-law tail with exponent 1")
    a1.legend(loc="lower left")

    w = 0.4
    games_share = freq / draws
    pay_share = payout / payout.sum()
    a2.bar(ks - w / 2 - 0.02, games_share, width=w, color=BLUE, label="Share of games", linewidth=0)
    a2.bar(ks + w / 2 + 0.02, pay_share, width=w, color=ORANGE, label="Share of money paid out",
           linewidth=0)
    a2.set_yscale("log")
    a2.set_xlabel("Flips until first heads (k)  →  payoff 2ᵏ")
    a2.set_ylabel("Share")
    a2.set_xticks(ks[::2])
    a2.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v * 100:.5f}".rstrip("0").rstrip(".") + "%"))
    a2.set_title("Rare games pay for everything")
    a2.set_ylim(top=12)
    a2.set_yticks([10.0 ** e for e in range(-7, 1)])
    a2.legend(loc="upper left", ncol=2)
    top_k = ks[np.argmax(pay_share)]
    a2.annotate(f"{freq[top_k - 1]:,} games\n= {pay_share[top_k - 1]:.0%} of all winnings",
                (top_k + w / 2, pay_share[top_k - 1]), xytext=(-40, 40),
                textcoords="offset points", color=INK_2, fontsize=9.5, ha="right",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    header(fig, "Why the expectation is infinite",
           "Each outcome k has probability 1/2ᵏ and pays 2ᵏ, so every outcome adds "
           "exactly $1 to the expected value, and there are infinitely many.\n"
           "Half the games pay $2, yet the payouts are spread roughly evenly "
           "across every k, all the way out to the one-in-ten-million games.", note)
    save(fig, outdir, "04_payoff_tail.png")


def plot_bankroll(outdir: Path, note: str) -> None:
    """Figure 05: expected value of one game when the casino can pay at most its bankroll."""
    bank = np.logspace(0.31, 15, 3000)
    ev = capped_ev(bank)

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.8, bottom=0.12)
    ax.fill_between(bank, 0, ev, color=BLUE, alpha=0.16, lw=0)
    ax.plot(bank, ev, color=BLUE, lw=2.4)

    marks = [
        (100, "Your wallet\n$100"),
        (1e6, "A millionaire\n$1M"),
        (1e9, "A big casino\n$1B"),
        (4e11, "Richest person\n~$400B"),
        (1.1e14, "World GDP\n~$110T"),
    ]
    for b, label in marks:
        v = float(capped_ev(b))
        ax.scatter([b], [v], s=60, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.annotate(f"{label}\nEV ${v:.2f}", (b, v), xytext=(0, 22),
                    textcoords="offset points", ha="center", va="bottom",
                    color=INK_2, fontsize=9.5, linespacing=1.3)

    ax.set_xscale("log")
    log_money_axis(ax.xaxis)
    ax.yaxis.set_major_formatter(FuncFormatter(dollars))
    ax.set_xlim(bank[0], bank[-1])
    ax.set_ylim(0, 62)
    ax.set_xlabel("Casino's bankroll (the most it can ever pay)")
    ax.set_ylabel("Expected value of one game")
    header(fig, "Infinity is expensive",
           "A real casino can't pay more than it has. Capping the payout turns ∞ "
           "into roughly log₂(bankroll) + 1,\nso even a casino backed by the "
           "entire world economy offers a game worth under $50.", note)
    save(fig, outdir, "05_finite_bankroll.png")


def plot_utility(outdir: Path, note: str) -> None:
    """Figure 06: certainty equivalent under log and square-root utility as wealth grows."""
    wealth = np.logspace(0, 9, 160)
    ce_log = np.array([certainty_equivalent(w, "log") for w in wealth])
    ce_sqrt = np.array([certainty_equivalent(w, "sqrt") for w in wealth])

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.08, right=0.83, top=0.8, bottom=0.12)
    ax.plot(wealth, ce_sqrt, color=ORANGE, lw=2.2, label="Square-root utility (Cramer, 1728)")
    ax.plot(wealth, ce_log, color=BLUE, lw=2.4, label="Log utility (Bernoulli, 1738)")
    for ce, dy, label in ((ce_log, -12, "Bernoulli\n(log)"), (ce_sqrt, 12, "Cramer\n(√)")):
        ax.annotate(label, (wealth[-1], ce[-1]), xytext=(10, dy), textcoords="offset points",
                    va="center", color=INK_2, fontsize=10, annotation_clip=False)

    for w in (1e2, 1e4, 1e6, 1e9):
        v = certainty_equivalent(w, "log")
        ax.scatter([w], [v], s=54, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.annotate(f"{money(w)} rich\npays ≤ ${v:.2f}", (w, v), xytext=(-8, 14),
                    textcoords="offset points", ha="right", color=INK_2, fontsize=9.5)

    ax.set_xscale("log")
    log_money_axis(ax.xaxis)
    ax.yaxis.set_major_formatter(FuncFormatter(dollars))
    ax.set_xlim(wealth[0], wealth[-1])
    ax.set_ylim(0, max(ce_sqrt.max(), ce_log.max()) * 1.12)
    ax.set_xlabel("Your total wealth")
    ax.set_ylabel("Most you'd rationally pay to play")
    ax.legend(loc="upper left")
    header(fig, "Bernoulli's way out",
           "If each extra dollar is worth a bit less to you than the last, the "
           "infinite game has a finite price. Under log utility\na billionaire "
           "pays about $31, close to what the simulations keep landing on.", note)
    save(fig, outdir, "06_utility_price.png")
