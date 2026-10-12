"""How the running average behaves as players keep playing."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

from ..formatting import count, money
from ..style import (AQUA, BLUE, BLUE_RAMP, DASHED, INK, INK_2, MAGENTA, MUTED, ORANGE,
                     SURFACE, colorbar, header, log_count_axis, log_money_axis, save,
                     spread_labels)


def plot_fan(n: np.ndarray, M: np.ndarray, outdir: Path, note: str) -> None:
    q = np.percentile(M, [1, 5, 25, 50, 75, 95, 99], axis=1)
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.08, right=0.83, top=0.8, bottom=0.12)

    for lo, hi, alpha, label in ((0, 6, 0.13, "1–99th pct"),
                                 (1, 5, 0.22, "5–95th pct"),
                                 (2, 4, 0.42, "25–75th pct")):
        ax.fill_between(n, q[lo], q[hi], color=BLUE, alpha=alpha, lw=0, label=label)
    ax.plot(n, q[3], color=BLUE_RAMP[200], lw=2.4, label="Median player", zorder=4)

    nn = n[n >= 2]
    ax.plot(nn, np.log2(nn), color=INK_2, lw=1.4, ls=DASHED,
            label="Feller's law: log₂ n", zorder=5)

    final = M[-1]
    picks = {
        "Luckiest player": (int(np.argmax(final)), ORANGE),
        "Typical player": (int(np.argsort(final)[len(final) // 2]), AQUA),
        "Unluckiest player": (int(np.argmin(final)), MAGENTA),
    }
    top = q[6].max() * 1.6
    ends = spread_labels([min(M[-1, idx], top * 0.8) for idx, _ in picks.values()], min_ratio=1.22)
    for (label, (idx, color)), y_text in zip(picks.items(), ends):
        ax.plot(n, M[:, idx], color=color, lw=1.6, alpha=0.95, zorder=6, label=label)
        y_end = min(M[-1, idx], top)
        ax.scatter([n[-1]], [y_end], s=46, color=color, edgecolor=SURFACE,
                   linewidth=2, zorder=7, clip_on=False)
        ax.annotate(f"{label.split()[0]}: ${M[-1, idx]:,.0f}", (n[-1], y_end),
                    xytext=(n[-1] * 1.6, y_text), textcoords="data", va="center",
                    color=INK_2, fontsize=9.5, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                                    shrinkA=0, shrinkB=4))

    ax.set_xscale("log")
    ax.set_yscale("log")
    log_count_axis(ax.xaxis)
    log_money_axis(ax.yaxis)
    ax.set_xlim(n[0], n[-1])
    ax.set_ylim(1.8, top)

    # The luckiest player's jackpot runs off the chart; label it rather than rescale.
    lucky = picks["Luckiest player"][0]
    totals = M[:, lucky] * n
    jump = int(np.argmax(np.diff(totals))) + 1
    if M[jump, lucky] > top:
        prize = money(float(f"{totals[jump] - totals[jump - 1]:.2g}"))
        ax.annotate(f"One game paid ~{prize}\naverage spikes to ${M[jump, lucky]:,.0f} ↑",
                    (n[jump], top), xytext=(-10, -8), textcoords="offset points",
                    ha="right", va="top", color=INK_2, fontsize=9.5)
    ax.set_xlabel("Games played (n)")
    ax.set_ylabel("Average payoff per game")
    ax.legend(loc="upper left", ncol=2)

    header(fig, "The average never settles",
           f"Running average payoff of {M.shape[1]:,} players, each playing "
           f"{count(n[-1])} games. The law of large numbers doesn't apply:\n"
           f"the typical average creeps up like log₂ n. After {count(n[-1])} games "
           f"the median player makes ${q[3, -1]:,.2f} a game.",
           note)
    save(fig, outdir, "01_running_mean_fan.png")


def plot_heatmap(n: np.ndarray, M: np.ndarray, outdir: Path, note: str) -> None:
    edges_y = np.linspace(np.log10(2), 3.2, 220)
    H = np.stack([np.histogram(np.log10(row), bins=edges_y)[0] for row in M], axis=1)
    H = H / H.max(axis=0, keepdims=True)

    log_n = np.log10(n)
    edges_x = np.concatenate([[log_n[0] - 0.01], (log_n[1:] + log_n[:-1]) / 2,
                              [log_n[-1] + 0.01]])
    cmap = LinearSegmentedColormap.from_list(
        "glow", [SURFACE, BLUE_RAMP[650], BLUE_RAMP[450], BLUE_RAMP[250], "#f4f8fe"])

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.08, right=0.86, top=0.8, bottom=0.12)
    mesh = ax.pcolormesh(10 ** edges_x, 10 ** edges_y, H ** 0.6, cmap=cmap,
                         shading="flat", rasterized=True)
    ax.grid(False)

    nn = n[n >= 2]
    ax.plot(nn, np.log2(nn), color=ORANGE, lw=2, ls=DASHED, label="log₂ n")
    ax.plot(n, np.median(M, axis=1), color=INK, lw=1.4, label="Median")

    ax.set_xscale("log")
    ax.set_yscale("log")
    log_count_axis(ax.xaxis)
    log_money_axis(ax.yaxis)
    ax.set_xlim(n[0], n[-1])
    ax.set_ylim(2, 10 ** 3.2)
    ax.set_xlabel("Games played (n)")
    ax.set_ylabel("Average payoff per game")
    ax.legend(loc="upper left")

    cb = colorbar(fig, mesh, [0.88, 0.12, 0.014, 0.68])
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["rare", "most\nplayers"])

    header(fig, f"Where {M.shape[1]:,} players' averages live",
           "Density of the running average across all players at each n "
           "(each column normalised). The bright ridge never stops climbing,\n"
           "and the long upward streaks are single jackpots that take thousands "
           "of games to wash out.", note)
    save(fig, outdir, "02_running_mean_density.png")


def plot_ridgeline(n: np.ndarray, M: np.ndarray, outdir: Path, note: str) -> None:
    decades = [10 ** e for e in range(1, int(np.log10(n[-1])) + 1)]
    ramp = [BLUE_RAMP[s] for s in (600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100)]
    ramp = ramp[-len(decades):] if len(decades) <= len(ramp) else ramp

    lo, hi = np.log10(2), 3.4
    xs = np.linspace(lo, hi, 600)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.12, right=0.95, top=0.83, bottom=0.1)

    gap, height = 1.0, 2.6
    for i, d in enumerate(decades):
        row = int(np.searchsorted(n, d))
        dens = gaussian_kde(np.log10(M[row]), bw_method=0.06)(xs)
        dens = dens / dens.max() * height
        base = (len(decades) - 1 - i) * gap
        z = 10 + i
        ax.fill_between(10 ** xs, base, base + dens, color=ramp[i % len(ramp)],
                        alpha=0.92, lw=0, zorder=z)
        ax.plot(10 ** xs, base + dens, color=SURFACE, lw=2.0, zorder=z + 0.1)
        ax.plot(10 ** xs, base + dens, color=INK, lw=0.7, alpha=0.55, zorder=z + 0.2)
        ax.text(1.75, base + 0.08, f"n = {count(d)}", color=INK_2, fontsize=10.5,
                ha="right", va="bottom", zorder=50)
        med = np.median(M[row])
        ax.plot([med, med], [base, base + np.interp(np.log10(med), xs, dens)],
                color=SURFACE, lw=1.4, zorder=z + 0.3)

    ax.set_xscale("log")
    log_money_axis(ax.xaxis)
    ax.set_xlim(2, 10 ** hi)
    ax.set_ylim(-0.1, (len(decades) - 1) * gap + height + 0.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Average payoff per game (across {M.shape[1]:,} players)")
    header(fig, "The distribution keeps moving right",
           "Each ridge is the spread of players' average winnings after n games. "
           "With a finite mean, these would pile up on one value.\nHere every "
           "10× more games shifts the whole crowd about $3.3 to the right, trailing a "
           "heavy tail of jackpot winners. Notch = median.", note)
    save(fig, outdir, "03_distribution_ridgeline.png")
