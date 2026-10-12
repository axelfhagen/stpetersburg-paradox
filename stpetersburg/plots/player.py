"""A single player's history, game by game."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from ..formatting import money, money3
from ..simulation import pick_journey
from ..style import (AXIS, BLUE, DASHED, INK_2, MUTED, ORANGE, RED, SURFACE, header, save)


def payoff_hist(ax, k: np.ndarray, kmax: int):
    """Horizontal bars counting how many games ended on each payoff. Returns (bars, labels)."""
    ks = np.arange(1, kmax + 1)
    counts = np.bincount(k, minlength=kmax + 1)[1:kmax + 1]
    bars = ax.barh(ks, counts, height=0.72, color=BLUE, edgecolor=SURFACE, linewidth=2)
    labels = [ax.text(c, kk, f"  {c:,}" if c else "", va="center", ha="left",
                      color=INK_2, fontsize=9) for kk, c in zip(ks, counts)]
    ax.set_yticks(ks)
    ax.set_yticklabels([money3(2.0 ** kk) for kk in ks], fontsize=9)
    ax.set_ylim(0.4, kmax + 0.6)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Number of games")
    return bars, labels


def journey_axes(fig, width_ratios, height_ratios, **grid_kw):
    """Three stacked time-series panels on the left, one tall histogram on the right."""
    gs = fig.add_gridspec(3, 2, width_ratios=width_ratios, height_ratios=height_ratios, **grid_kw)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[1, 0], sharex=a1)
    a3 = fig.add_subplot(gs[2, 0], sharex=a1)
    ah = fig.add_subplot(gs[:, 1])
    return a1, a2, a3, ah


def biggest_games(pay: np.ndarray, limit: int = 3) -> list[int]:
    """Indices of up to `limit` big jackpots that are spread out enough to label separately."""
    games = len(pay)
    picked: list[int] = []
    for i in np.argsort(pay)[::-1]:
        if len(picked) == limit or pay[i] < pay.max() / 8:
            break
        if all(abs(i - j) > games * 0.06 for j in picked):
            picked.append(int(i))
    return picked


def plot_journey(outdir: Path, rng: np.random.Generator, games: int = 3000,
                 price: float = 15.0) -> None:
    k, pay = pick_journey(rng, games)
    x = np.arange(1, games + 1)
    run = np.cumsum(pay) / x
    profit = np.cumsum(pay) - price * x
    kmax = int(k.max())
    top3 = biggest_games(pay)

    fig = plt.figure(figsize=(15, 9.6))
    a1, a2, a3, ah = journey_axes(fig, [3, 1.05], [1.35, 1, 1], left=0.07, right=0.97,
                                  top=0.8, bottom=0.08, hspace=0.5, wspace=0.16)

    a1.vlines(x, 0, pay, color=BLUE, lw=0.9)
    a1.set_ylim(0, pay.max() * 1.3)
    a1.yaxis.set_major_formatter(FuncFormatter(money))
    a1.set_title("What each game paid")
    for i in top3:
        a1.annotate(f"${pay[i]:,.0f}\n{k[i] - 1} tails in a row", (x[i], pay[i]),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                    color=INK_2, fontsize=9, linespacing=1.2)
    small = np.mean(pay <= 8)
    a1.text(0.01, 0.93, f"{small:.0%} of games paid $8 or less,\nthe flat strip along the bottom",
            transform=a1.transAxes, color=MUTED, fontsize=9.5, va="top")

    a2.plot(x, run, color=ORANGE, lw=1.8, label="Your running average")
    a2.plot(x, np.log2(x), color=INK_2, lw=1.2, ls=DASHED, label="log₂ n (typical player)")
    a2.axhline(price, color=MUTED, lw=1, ls=":")
    a2.text(games * 0.995, price, f"ticket ${price:g}", color=MUTED, fontsize=9,
            ha="right", va="bottom")
    a2.set_ylim(0, run[20:].max() * 1.15)
    a2.yaxis.set_major_formatter(FuncFormatter(money))
    a2.set_title("Average payoff so far")
    a2.legend(loc="upper left", ncol=2)

    a3.fill_between(x, 0, profit, where=profit >= 0, color=BLUE, alpha=0.35, lw=0, interpolate=True)
    a3.fill_between(x, 0, profit, where=profit < 0, color=RED, alpha=0.35, lw=0, interpolate=True)
    a3.plot(x, profit, color=INK_2, lw=1.3)
    a3.axhline(0, color=AXIS, lw=1)
    a3.yaxis.set_major_formatter(FuncFormatter(money))
    a3.set_title(f"Your bank balance at ${price:g} a game")
    a3.set_xlabel("Game number")
    a3.set_xlim(0, games)
    worst = int(np.argmin(profit))
    a3.annotate(f"Down {money3(-profit[worst])}", (x[worst], profit[worst]),
                xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                color=INK_2, fontsize=9)

    for ax in (a1, a2, a3):
        for i in top3:
            ax.axvline(x[i], color=MUTED, lw=0.8, ls=":", alpha=0.6, zorder=0)
        plt.setp(ax.get_xticklabels(), visible=ax is a3)

    payoff_hist(ah, k, kmax)
    ah.set_xlim(0, (k == 1).sum() * 1.35)
    ah.set_title("How often each payoff came up")
    mean_k = np.log2(run[-1])
    ah.axhline(mean_k, color=ORANGE, lw=1.6)
    ah.text(ah.get_xlim()[1] * 0.97, mean_k + 0.25,
            f"Your average: ${run[-1]:.2f}\n(no single game can pay this)",
            color=INK_2, fontsize=9.5, ha="right", va="bottom")

    jackpot_share = pay[top3].sum() / pay.sum()
    header(fig, f"One player, {games:,} games",
           f"Almost every game pays pocket change. Then a long run of tails pays thousands: the "
           f"{len(top3)} biggest games are {jackpot_share:.0%} of everything this player won.\n"
           "The average lurches up on each jackpot, then slowly bleeds back down "
           "until the next one.", "Each game: flip until heads; k flips pays 2ᵏ.")
    save(fig, outdir, "07_one_player_journey.png")
