"""Animated versions of the main ideas, saved as GIF (and MP4 if ffmpeg is installed)."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter

from .formatting import money, percent
from .plots.player import journey_axes, payoff_hist
from .simulation import capped_ev, pick_journey
from .style import (AXIS, BLUE, DASHED, INK, INK_2, MUTED, ORANGE, RED, SERIES, SURFACE,
                    header, log_count_axis, log_money_axis)


def save_anim(anim, fig, outdir: Path, stem: str, fps: int) -> None:
    """Write the animation as GIF, plus MP4 when ffmpeg is on the PATH, then close the figure."""
    gif = outdir / f"{stem}.gif"
    anim.save(gif, writer=animation.PillowWriter(fps=fps), dpi=110,
              savefig_kwargs={"facecolor": SURFACE})
    print(f"  {gif}")
    if shutil.which("ffmpeg"):
        mp4 = outdir / f"{stem}.mp4"
        writer = animation.FFMpegWriter(fps=fps, bitrate=6000, extra_args=["-pix_fmt", "yuv420p"])
        anim.save(mp4, writer=writer, dpi=160, savefig_kwargs={"facecolor": SURFACE})
        print(f"  {mp4}")
    plt.close(fig)


def counter_text(fig):
    """Empty top-right text artist, updated each frame with the game count."""
    return fig.text(0.955, 0.955, "", color=INK, fontsize=19, fontweight="bold",
                    ha="right", va="top")


def anim_race(outdir: Path, rng: np.random.Generator, players: int = 5,
              games: int = 300_000, frames: int = 220, hold: int = 45, fps: int = 30) -> None:
    """anim_01: running averages of a few players on log axes, with callouts for big jackpots."""
    k = rng.geometric(0.5, size=(players, games))
    payoff = 2.0 ** k
    run = np.cumsum(payoff, axis=1) / np.arange(1, games + 1)
    colors = SERIES[:players]

    # Sample each path on a log grid, plus every big jackpot and the game before it,
    # so the spikes survive the downsampling.
    base_idx = np.round(np.logspace(0, np.log10(games), 3000)).astype(int) - 1
    paths = []
    for p in range(players):
        big = np.flatnonzero(payoff[p] >= 2 ** 9)
        idx = np.unique(np.concatenate([base_idx, big, np.maximum(big - 1, 0)]))
        paths.append((idx + 1, run[p, idx]))

    # The dozen biggest jackpots get a callout when they land.
    flat = np.argsort(payoff, axis=None)[-12:]
    events = [(int(i % games) + 1, int(i // games), payoff.flat[i]) for i in flat]

    heads = np.unique(np.round(np.logspace(0, np.log10(games), frames)).astype(int))
    timeline = np.concatenate([heads, np.full(hold, heads[-1])])

    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.8, bottom=0.12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_count_axis(ax.xaxis)
    log_money_axis(ax.yaxis)
    ax.set_xlim(1, games * 1.05)
    ax.set_ylim(1.8, run.max() * 1.6)
    ax.set_xlabel("Games played")
    ax.set_ylabel("Average payoff per game")

    g = np.logspace(np.log10(2), np.log10(games), 200)
    ax.plot(g, np.log2(g), color=INK_2, lw=1.2, ls=DASHED, alpha=0.8)
    ax.text(games * 0.9, np.log2(games * 0.9) * 0.82, "log₂ n", color=MUTED,
            fontsize=10, ha="right", va="top")

    lines = [ax.plot([], [], color=c, lw=1.8)[0] for c in colors]
    dots = [ax.scatter([], [], s=50, color=c, edgecolor=SURFACE, linewidth=2, zorder=6)
            for c in colors]
    rings = ax.scatter([], [], s=[], facecolor="none", edgecolor=[], linewidth=1.8, zorder=7)
    callouts = [ax.text(0, 0, "", color=INK, fontsize=9.5, fontweight="bold",
                        ha="center", va="bottom", zorder=8) for _ in events]

    board = [fig.text(0.115, 0.73 - 0.042 * p, "", color=INK_2, fontsize=10.5,
                      family=["Menlo", "DejaVu Sans Mono"]) for p in range(players)]
    swatches = [fig.add_artist(plt.Rectangle((0.1, 0.733 - 0.042 * p), 0.009, 0.016,
                                             transform=fig.transFigure, color=colors[p]))
                for p in range(players)]
    counter = counter_text(fig)
    header(fig, "Five players, one infinite game",
           "Running average payoff per game. Each jump is a single lucky streak of tails. "
           "Afterwards the average decays,\nbut never quite back down.")

    land_frame = {e: int(np.searchsorted(timeline[:len(heads)], e[0])) for e in events}

    def update(f):
        h = timeline[f]
        for p, (xs, ys) in enumerate(paths):
            c = int(np.searchsorted(xs, h, side="right"))
            lines[p].set_data(xs[:c], ys[:c])
            dots[p].set_offsets([[h, run[p, h - 1]]])
        order = np.argsort(-run[:, h - 1])
        for rank, p in enumerate(order):
            board[rank].set_text(f"Player {p + 1}  {'$' + format(run[p, h - 1], ',.2f'):>10}")
            swatches[rank].set_color(colors[p])
        counter.set_text(f"{h:,} games")

        offs, sizes, edges = [], [], []
        for t, e in zip(callouts, events):
            n_e, p_e, prize = e
            age = f - land_frame[e]
            if n_e <= h and 0 <= age < 30:
                fade = 1 - age / 30
                y = run[p_e, n_e - 1]
                offs.append([n_e, y])
                sizes.append(80 + age * 14)
                edges.append((*mcolors.to_rgb(colors[p_e]), fade))
                t.set_position((n_e, y * 1.25))
                t.set_text(f"+${prize:,.0f}")
                t.set_alpha(fade)
            else:
                t.set_text("")
        rings.set_offsets(np.array(offs) if offs else np.empty((0, 2)))
        rings.set_sizes(sizes)
        rings.set_edgecolors(edges if edges else "none")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(timeline), blit=False)
    save_anim(anim, fig, outdir, "anim_01_player_race", fps)


def anim_distributions(n: np.ndarray, M: np.ndarray, Mc: np.ndarray, cap: float,
                       outdir: Path, frames: int = 180, hold: int = 45, fps: int = 30) -> None:
    """anim_02: distribution of averages over time, unlimited (`M`) vs. capped bankroll (`Mc`)."""
    rows = np.unique(np.linspace(0, len(n) - 1, frames).round().astype(int))
    timeline = np.concatenate([rows, np.full(hold, rows[-1])])
    bins = np.logspace(np.log10(2), 3.6, 140)
    ev_cap = float(capped_ev(cap))

    def hist(row, data):
        h, _ = np.histogram(data[row], bins=bins)
        return h / data.shape[1]

    ymax_top = max(hist(r, M).max() for r in rows) * 1.15
    ymax_bot = max(hist(r, Mc).max() for r in rows) * 1.08

    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.8, bottom=0.1, hspace=0.38)
    for ax, ymax, title in ((top, ymax_top, "Unlimited bank: the textbook game"),
                            (bot, ymax_bot, f"Casino bankroll capped at {money(cap)}")):
        ax.set_xscale("log")
        log_money_axis(ax.xaxis)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_formatter(FuncFormatter(percent))
        ax.set_ylabel("Share of players")
        ax.set_title(title)
    bot.set_xlabel(f"Average payoff per game, across {M.shape[1]:,} players")

    step_top = top.stairs(hist(0, M), bins, fill=True, color=BLUE, alpha=0.85)
    step_bot = bot.stairs(hist(0, Mc), bins, fill=True, color=ORANGE, alpha=0.85)
    bot.axvline(ev_cap, color=INK_2, lw=1.2, ls=DASHED)
    bot.text(ev_cap * 1.08, ymax_bot * 0.92, f"EV = ${ev_cap:.2f}", color=INK_2,
             fontsize=10, va="top")
    feller = top.axvline(2, color=INK_2, lw=1.2, ls=DASHED)
    feller_txt = top.text(2, ymax_top * 0.92, "", color=INK_2, fontsize=10, va="top")
    med_top = top.axvline(2, color=INK, lw=1.6)
    med_bot = bot.axvline(2, color=INK, lw=1.6)
    counter = counter_text(fig)
    header(fig, "Infinite vs. finite: one converges",
           "Distribution of players' average winnings as everyone keeps playing. "
           "A finite bank gives a finite mean,\nand the crowd collapses onto it. "
           "The textbook game never settles; it just drifts right. White line = median.")

    def update(f):
        r = timeline[f]
        step_top.set_data(hist(r, M))
        step_bot.set_data(hist(r, Mc))
        m1, m2 = np.median(M[r]), np.median(Mc[r])
        med_top.set_xdata([m1, m1])
        med_bot.set_xdata([m2, m2])
        lg = max(np.log2(n[r]), 2)
        feller.set_xdata([lg, lg])
        feller_txt.set_position((lg * 1.08, ymax_top * 0.92))
        feller_txt.set_text(f"log₂ n = {lg:.1f}")
        counter.set_text(f"{n[r]:,} games")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(timeline), blit=False)
    save_anim(anim, fig, outdir, "anim_02_infinite_vs_finite", fps)


def anim_journey(outdir: Path, rng: np.random.Generator, games: int = 2000,
                 price: float = 12.0, frames: int = 240, hold: int = 60, fps: int = 30) -> None:
    """anim_03: one player's game-by-game history, revealed progressively (animated figure 07)."""
    k, pay = pick_journey(rng, games)
    x = np.arange(1, games + 1)
    run = np.cumsum(pay) / x
    profit = np.cumsum(pay) - price * x
    kmax = int(k.max())
    heads = np.unique(np.round(np.linspace(1, games, frames)).astype(int))
    timeline = np.concatenate([heads, np.full(hold, games)])
    events = np.flatnonzero(pay >= 2 ** 10)
    land = {int(e): int(np.searchsorted(heads, e + 1)) for e in events}

    fig = plt.figure(figsize=(12.5, 7.6))
    a1, a2, a3, ah = journey_axes(fig, [2.7, 1], [1.3, 1, 1], left=0.08, right=0.97,
                                  top=0.8, bottom=0.08, hspace=0.55, wspace=0.18)

    segs = np.stack([np.column_stack([x, np.zeros(games)]), np.column_stack([x, pay])], axis=1)
    spikes = LineCollection([], colors=BLUE, linewidths=1.0)
    a1.add_collection(spikes)
    a1.set_title("What each game paid")
    a1.yaxis.set_major_formatter(FuncFormatter(money))
    callouts = {int(e): a1.text(0, 0, "", color=INK, fontsize=10, fontweight="bold",
                                ha="center", va="bottom") for e in events}

    (l_run,) = a2.plot([], [], color=ORANGE, lw=1.8, label="Running average")
    a2.plot(x, np.log2(x), color=INK_2, lw=1.1, ls=DASHED, label="log₂ n")
    a2.axhline(price, color=MUTED, lw=1, ls=":")
    a2.set_title("Average payoff so far")
    a2.yaxis.set_major_formatter(FuncFormatter(money))
    a2.legend(loc="lower right", ncol=2)
    avg_txt = a2.text(0.01, 0.9, "", transform=a2.transAxes, color=INK, fontsize=11,
                      fontweight="bold", va="top")

    (l_pro,) = a3.plot([], [], color=INK_2, lw=1.2)
    a3.axhline(0, color=AXIS, lw=1)
    a3.set_title(f"Bank balance at ${price:g} a game")
    a3.set_xlabel("Game number")
    a3.yaxis.set_major_formatter(FuncFormatter(money))
    a3.set_xlim(0, games)
    fills = []
    for ax in (a1, a2):
        plt.setp(ax.get_xticklabels(), visible=False)

    bars, labels = payoff_hist(ah, k, kmax)
    ah.set_xlim(0, (k == 1).sum() * 1.4)
    ah.set_title("Payoff tally")
    counter = fig.text(0.97, 0.955, "", color=INK, fontsize=19, fontweight="bold",
                       ha="right", va="top")
    header(fig, "Pocket change, pocket change, jackpot",
           "One player, game by game. Most games pay $2 or $4; rarely a long run of tails "
           "pays thousands,\nwhich yanks up the average and the bank balance before the "
           "slow bleed resumes.")

    def update(f):
        h = timeline[f]
        spikes.set_segments(segs[:h])
        a1.set_ylim(0, max(pay[:h].max() * 1.3, 16))
        l_run.set_data(x[:h], run[:h])
        a2.set_ylim(0, max(run[min(h, 10) - 1:h].max(), np.log2(h), price) * 1.25)
        avg_txt.set_text(f"${run[h - 1]:.2f}")
        l_pro.set_data(x[:h], profit[:h])
        lo, hi = profit[:h].min(), profit[:h].max()
        pad = max(hi - lo, 20) * 0.15
        a3.set_ylim(min(lo, 0) - pad, max(hi, 0) + pad)
        for coll in fills:
            coll.remove()
        fills.clear()
        fills.append(a3.fill_between(x[:h], 0, profit[:h], where=profit[:h] >= 0, color=BLUE,
                                     alpha=0.35, lw=0, interpolate=True))
        fills.append(a3.fill_between(x[:h], 0, profit[:h], where=profit[:h] < 0, color=RED,
                                     alpha=0.35, lw=0, interpolate=True))
        counts = np.bincount(k[:h], minlength=kmax + 1)[1:kmax + 1]
        for bar, lab, c in zip(bars, labels, counts):
            bar.set_width(c)
            lab.set_x(c)
            lab.set_text(f"  {c:,}" if c else "")
        last_x, lift = -games, 0.0
        for e, t in sorted(callouts.items()):
            age = f - land[e]
            if e < h and 0 <= age < 45:
                # Stack callouts for jackpots that land close together.
                lift = lift + 0.12 if e - last_x < games * 0.05 else 0.0
                last_x = e
                t.set_position((x[e], pay[e] + lift * a1.get_ylim()[1]))
                t.set_text(f"+${pay[e]:,.0f}")
                t.set_alpha(1 - age / 45)
            else:
                t.set_text("")
        counter.set_text(f"Game {h:,}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(timeline), blit=False)
    save_anim(anim, fig, outdir, "anim_03_one_player_journey", fps)
