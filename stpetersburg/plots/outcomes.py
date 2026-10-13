"""What the game means for real players: profit odds, inequality, and a fair price."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullFormatter

from ..formatting import count, dollars, money, percent, words
from ..simulation import batch_sums
from ..style import (AXIS, BLUE, BLUE_RAMP, INK, INK_2, MUTED, ORANGE, RED, SURFACE, colorbar,
                     header, log_count_axis, save)


def plot_profit_odds(n: np.ndarray, M: np.ndarray, outdir: Path, note: str) -> None:
    """Figure 08: share of players in profit for each ticket price and number of games."""
    prices = np.linspace(2, 50, 241)
    P = np.empty((len(n), len(prices)))
    for i, row in enumerate(M):
        srt = np.sort(row)
        P[i] = 1 - np.searchsorted(srt, prices, side="right") / len(srt)

    log_n = np.log10(n)
    edges_x = 10 ** np.concatenate([[log_n[0]], (log_n[1:] + log_n[:-1]) / 2, [log_n[-1]]])
    step = prices[1] - prices[0]
    edges_y = np.concatenate([prices - step / 2, [prices[-1] + step / 2]])
    cmap = LinearSegmentedColormap.from_list(
        "ahead", [RED, "#b0504f", "#383835", "#2f6cb8", BLUE_RAMP[250]])

    fig, ax = plt.subplots(figsize=(12, 6.9))
    fig.subplots_adjust(left=0.08, right=0.86, top=0.79, bottom=0.12)
    mesh = ax.pcolormesh(edges_x, edges_y, P.T, cmap=cmap, vmin=0, vmax=1,
                         shading="flat", rasterized=True)
    ax.grid(False)
    ax.contour(n, prices, P.T, levels=[0.5, 0.9], colors=INK, linewidths=[1.8, 0.9])
    for level, at, dy, label in ((0.5, 1e3, 11, "half of players ahead"),
                                 (0.9, 3e7, -12, "90% ahead")):
        i = int(np.searchsorted(n, at))
        j = int(np.argmin(np.abs(P[i] - level)))
        ax.annotate(label, (n[i], prices[j]), xytext=(0, dy), textcoords="offset points",
                    ha="center", va="center", color=INK, fontsize=9.5, rotation=17)

    for price in (10, 20, 30):
        j = int(np.searchsorted(prices, price))
        cross = np.flatnonzero(P[:, j] >= 0.5)
        if cross.size:
            nn = n[cross[0]]
            ax.scatter([nn], [price], s=60, color=INK, edgecolor=SURFACE, linewidth=2, zorder=5)
            ax.annotate(f"${price} ticket: most players\nahead after ~{count(float(f'{nn:.1g}'))} games",
                        (nn, price), xytext=(-12, 8), textcoords="offset points", ha="right",
                        color=INK, fontsize=9.5)

    ax.set_xscale("log")
    log_count_axis(ax.xaxis)
    ax.yaxis.set_major_formatter(FuncFormatter(dollars))
    ax.set_xlim(n[0], n[-1])
    ax.set_xlabel("Games played (n)")
    ax.set_ylabel("Price per game")
    cb = colorbar(fig, mesh, [0.88, 0.12, 0.014, 0.67])
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cb.ax.yaxis.set_major_formatter(FuncFormatter(percent))
    cb.set_label("Chance you're in profit", color=INK_2)

    header(fig, "Will you come out ahead?",
           "Chance a player who pays $p per game is in profit after n games. Every price "
           "eventually wins, since the 50% line climbs forever,\nbut each extra dollar on the "
           "ticket doubles how long you must play. The fair price depends on how long you play.",
           note)
    save(fig, outdir, "08_chance_of_profit.png")


def plot_who_wins(n: np.ndarray, M: np.ndarray, Mc: np.ndarray, cap: float,
                  outdir: Path, note: str) -> None:
    """Figure 09: share of total winnings held by the luckiest player, the rest of the top 1%
    and everyone else, for the unlimited game and the bankroll-capped game (`Mc`).
    """
    decades = [d for d in (1, 100, 10 ** 4, 10 ** 6, 10 ** 9) if d <= n[-1]]
    players = M.shape[1]
    top = max(players // 100, 2)

    def shares(data, d):
        totals = np.sort(data[int(np.searchsorted(n, d))] * d)[::-1]
        whole = totals.sum()
        return totals[0] / whole, totals[1:top].sum() / whole, totals[top:].sum() / whole

    rows = np.array([shares(M, d) for d in decades] + [shares(Mc, decades[-1])])
    labels = [f"After {count(d)} game{'s' if d > 1 else ''}" for d in decades]
    labels.append(f"{money(cap)}-capped casino,\nafter {count(decades[-1])} games")
    top1 = rows[1:len(decades), :2].sum(axis=1)

    segments = [("The single luckiest player", BLUE_RAMP[100], SURFACE),
                (f"Next {top - 1} luckiest (rest of top 1%)", BLUE_RAMP[350], SURFACE),
                ("Everyone else (99%)", BLUE_RAMP[600], INK)]
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(left=0.19, right=0.96, top=0.75, bottom=0.11)
    y = np.arange(len(rows))[::-1].astype(float)
    y[-1] -= 0.35  # set the capped-casino row apart
    left = np.zeros(len(rows))
    for j, (label, color, ink) in enumerate(segments):
        w = rows[:, j]
        ax.barh(y, w, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2,
                label=label)
        for yi, li, wi in zip(y, left, w):
            if wi > 0.04:
                ax.text(li + wi / 2, yi, f"{wi:.0%}", ha="center", va="center", color=ink,
                        fontsize=11, fontweight="bold")
        left += w
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.axhline(y[-1] + 0.62, color=AXIS, lw=1, ls=(0, (2, 3)))
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(FuncFormatter(percent))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of all money the casino paid out")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    header(fig, "The money never evens out",
           f"{players:,} players each play the same number of games. In a normal game, playing "
           f"longer evens everyone out, so the top 1% hold ~1%.\nHere the top 1% keep "
           f"{top1.min():.0%}–{top1.max():.0%} of all winnings however long everyone plays. "
           "Cap the casino's bankroll and it behaves normally again.", note)
    save(fig, outdir, "09_who_gets_the_money.png")


def plot_worth(outdir: Path, rng: np.random.Generator, players: int, note: str) -> None:
    """Figure 10: spread of average payoff for real-world numbers of games, against log2 n."""
    scenarios = [
        ("Play once", 1),
        ("An evening at the table", 100),
        ("A game every minute for a year", 525_600),
        ("A game every second of an 80-year life", 2_524_608_000),
        ("Every human plays once", 8_100_000_000),
        ("Every human, a game a second, for a day", 8_100_000_000 * 86_400),
        ("A game every second since the Big Bang", 435_000_000_000_000_000),
    ]
    stats = []
    for _, m in scenarios:
        avg = batch_sums(np.full(players, m, dtype=np.int64), rng) / m
        stats.append(np.percentile(avg, [5, 25, 50, 75, 95]))
    stats = np.array(stats)

    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.subplots_adjust(left=0.33, right=0.84, top=0.79, bottom=0.11)
    y = np.arange(len(scenarios))[::-1]
    ax.hlines(y, stats[:, 0], stats[:, 4], color=BLUE_RAMP[500], lw=2.2)
    ax.hlines(y, stats[:, 1], stats[:, 3], color=BLUE, lw=11)
    ax.scatter(stats[:, 2], y, s=90, color=INK, edgecolor=BLUE_RAMP[600], linewidth=2, zorder=5,
               label="Typical player (median)")
    feller = np.array([np.log2(m) if m > 1 else np.nan for _, m in scenarios])
    ax.scatter(feller, y, s=60, marker="D", facecolor="none", edgecolor=ORANGE, linewidth=1.6,
               zorder=6, label="Feller's prediction: log₂ n")
    for yi, st in zip(y, stats):
        ax.text(st[4] * 1.12, yi, f"${st[2]:,.2f}", va="center", color=INK, fontsize=10.5,
                fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(y))
    for yi, (label, m) in zip(y, scenarios):
        ax.text(-0.02, yi + 0.12, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", color=INK, fontsize=11)
        ax.text(-0.02, yi - 0.2, f"{words(m)} game{'s' if m > 1 else ''}",
                transform=ax.get_yaxis_transform(), ha="right", va="center", color=MUTED,
                fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50, 100, 200])
    ax.xaxis.set_major_formatter(FuncFormatter(money))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(1.5, stats[:, 4].max() * 2.2)
    ax.set_ylim(-0.7, len(y) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average payoff per game  (bar: middle 50% of players · line: 5–95%)")
    ax.legend(loc="lower left")
    ax.annotate("Textbook\nexpected value:\n∞  →", xy=(1.0, 0.5), xycoords="axes fraction",
                xytext=(14, 0), textcoords="offset points", ha="left", va="center",
                color=ORANGE, fontsize=12, fontweight="bold", annotation_clip=False)
    header(fig, "What is the game actually worth to you?",
           "The textbook answer is infinity. What you actually average depends on how often "
           "you get to play. Even playing\nevery second since the Big Bang, the typical "
           f"player averages about ${stats[-1, 2]:.0f} a game.", note)
    save(fig, outdir, "10_what_its_worth_to_you.png")


def plot_gamblers(n: np.ndarray, M: np.ndarray, outdir: Path, note: str,
                  price: float = 25.0, shown: int = 300) -> None:
    """Figure 11: cumulative profit of `shown` players who pay `price` per game."""
    profit = n[:, None] * (M - price)
    ahead = (M > price).mean(axis=1)
    sample = profit[:, :shown]

    fig, ax = plt.subplots(figsize=(12, 6.9))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.79, bottom=0.12)
    ax.plot(n, np.where(sample < 0, sample, np.nan), color=RED, lw=0.7, alpha=0.3)
    ax.plot(n, np.where(sample >= 0, sample, np.nan), color=BLUE, lw=0.7, alpha=0.3)
    ax.plot(n, np.median(profit, axis=1), color=INK, lw=2.4, label="Median player")
    ax.plot([], [], color=BLUE, lw=2, label="In profit")
    ax.plot([], [], color=RED, lw=2, label="In the red")
    ax.axhline(0, color=INK_2, lw=1)

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=100)
    log_count_axis(ax.xaxis)
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.set_xlim(n[0], n[-1])
    lim = np.percentile(np.abs(sample), 99.5) * 3
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Games played (n)")
    ax.set_ylabel("Profit (symmetric log scale)")
    ax.legend(loc="upper left")

    for d in (10 ** 3, 10 ** 6, 10 ** 9):
        if d <= n[-1]:
            i = int(np.searchsorted(n, d))
            pct = f"{ahead[i]:.1%}" if 0.99 < ahead[i] < 1 else f"{ahead[i]:.0%}"
            ax.annotate(f"{pct} of players\nahead", (d, 0), xytext=(0, -28),
                        textcoords="offset points", ha="right" if d == n[-1] else "center",
                        va="top", color=INK, fontsize=10, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=AXIS, lw=0.8))

    header(fig, f"Pay ${price:g} a game and wait",
           f"Profit of {shown} players who each pay ${price:g} per game. For millions of games "
           "almost everyone bleeds money: costs grow like n,\nwinnings like n·log₂ n. "
           "But log₂ n eventually beats any fixed price, so in the very long run nearly "
           "everyone wins.", note)
    save(fig, outdir, "11_pay_and_wait.png")
