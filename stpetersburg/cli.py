"""Command-line entry point: run the simulations and render every figure."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from . import animations, plots  # noqa: E402
from .formatting import count, money  # noqa: E402
from .simulation import make_checkpoints, running_means  # noqa: E402
from .style import apply_theme  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line options (see `python -m stpetersburg --help`)."""
    ap = argparse.ArgumentParser(
        prog="stpetersburg",
        description="Simulate the St. Petersburg game and render plots and animations.")
    ap.add_argument("--paths", type=int, default=10_000, help="number of simulated players")
    ap.add_argument("--max-exp", type=int, default=9, help="each player plays 10^max_exp games")
    ap.add_argument("--cap", type=float, default=1e6,
                    help="bankroll for the finite-casino comparison")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--no-anim", action="store_true", help="skip GIF/MP4 rendering")
    ap.add_argument("--only", nargs="+", default=None,
                    help="render only outputs whose prefix matches, e.g. --only 07 anim_03")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    """Run both simulations once, then render the selected figures and animations."""
    args = parse_args(argv)
    apply_theme()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n = make_checkpoints(args.max_exp)
    note = (f"Simulation: {args.paths:,} players × {count(n[-1])} games "
            f"({args.paths * float(n[-1]):.0e} games total, exact via binomial splitting) "
            f"· seed {args.seed}")

    t0 = time.time()
    print(f"Simulating {args.paths:,} players x {n[-1]:,} games")
    M = running_means(args.paths, n, rng)
    print(f"Simulating capped casino ({money(args.cap)})")
    Mc = running_means(args.paths, n, rng, cap=args.cap)
    print(f"Done in {time.time() - t0:.1f}s\n\nRendering plots")

    def sub(stream: int) -> np.random.Generator:
        # A separate RNG per output, so --only reproduces the full run exactly.
        return np.random.default_rng([args.seed, stream])

    out, cap = args.out, args.cap
    worth_note = (f"Each scenario: {args.paths:,} players, exact sums via binomial splitting "
                  f"· seed {args.seed}")
    figures = [
        ("01", lambda: plots.plot_fan(n, M, out, note)),
        ("02", lambda: plots.plot_heatmap(n, M, out, note)),
        ("03", lambda: plots.plot_ridgeline(n, M, out, note)),
        ("04", lambda: plots.plot_tail(
            out, sub(4), "Simulation: 20M individual games, each flipped coin by coin")),
        ("05", lambda: plots.plot_bankroll(
            out, "EV with bankroll B: ⌊log₂ B⌋ + B / 2^⌊log₂ B⌋")),
        ("06", lambda: plots.plot_utility(
            out, "Certainty equivalent: the price p where E[u(wealth − p + X)] = u(wealth)")),
        ("07", lambda: plots.plot_journey(out, sub(7))),
        ("08", lambda: plots.plot_profit_odds(n, M, out, note)),
        ("09", lambda: plots.plot_who_wins(n, M, Mc, cap, out, note)),
        ("10", lambda: plots.plot_worth(out, sub(10), args.paths, worth_note)),
        ("11", lambda: plots.plot_gamblers(n, M, out, note)),
    ]
    anims = [
        ("anim_01", lambda: animations.anim_race(out, sub(101))),
        ("anim_02", lambda: animations.anim_distributions(n, M, Mc, cap, out)),
        ("anim_03", lambda: animations.anim_journey(out, sub(103))),
    ]

    def wanted(key: str) -> bool:
        return not args.only or any(key.startswith(o) for o in args.only)

    for key, render in figures:
        if wanted(key):
            render()
    if not args.no_anim and any(wanted(key) for key, _ in anims):
        print("\nRendering animations")
        for key, render in anims:
            if wanted(key):
                render()

    print(f"\nFinished in {time.time() - t0:.1f}s, output in {args.out.resolve()}")
