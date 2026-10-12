"""Colours, matplotlib theme and shared figure helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from .formatting import count, money

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#8a897f"
GRID = "#2e2e2b"
AXIS = "#4a4945"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#3987e5", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
)
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# Light to dark
BLUE_RAMP = {
    100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef",
    300: "#6da7ec", 350: "#5598e7", 400: "#3987e5", 450: "#2a78d6",
    500: "#256abf", 550: "#1c5cab", 600: "#184f95", 650: "#104281",
    700: "#0d366b",
}

DASHED = (0, (4, 3))


def _fonts() -> list[str]:
    # DejaVu Sans goes last as a per-glyph fallback for characters like ₂ and ⌊.
    available = {f.name for f in fm.fontManager.ttflist}
    preferred = [n for n in ("Inter", "Avenir Next", "Helvetica Neue") if n in available][:1]
    return preferred + ["DejaVu Sans"]


def apply_theme() -> None:
    plt.rcParams.update({
        "font.family": _fonts(),
        "text.parse_math": False,  # "$2 or $4" is money, not LaTeX
        "font.size": 11,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
    })


def header(fig, title: str, subtitle: str, note: str | None = None) -> None:
    fig.text(0.055, 0.955, title, color=INK, fontsize=19, fontweight="bold",
             ha="left", va="top")
    fig.text(0.055, 0.905, subtitle, color=INK_2, fontsize=11.5, ha="left", va="top")
    if note:
        fig.text(0.055, 0.025, note, color=MUTED, fontsize=8.5, ha="left", va="bottom")


def log_money_axis(axis) -> None:
    axis.set_major_locator(LogLocator(base=10))
    axis.set_major_formatter(FuncFormatter(money))
    axis.set_minor_formatter(NullFormatter())


def log_count_axis(axis) -> None:
    axis.set_major_locator(LogLocator(base=10))
    axis.set_major_formatter(FuncFormatter(count))
    axis.set_minor_formatter(NullFormatter())


def colorbar(fig, mesh, rect: list[float]):
    cb = fig.colorbar(mesh, cax=fig.add_axes(rect))
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, labelsize=9)
    return cb


def spread_labels(ys, min_ratio: float) -> list[float]:
    """Push label y-positions (log scale) apart so no two are closer than min_ratio."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        out[b] = max(out[b], out[a] * min_ratio)
    return list(out)


def save(fig, outdir: Path, name: str) -> None:
    path = outdir / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  {path}")
