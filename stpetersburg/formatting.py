"""Number formatting for axis ticks and labels."""

from __future__ import annotations


def money(x: float, _=None) -> str:
    """Dollar amount with a k/M/B/T/Q suffix. The unused second argument fits FuncFormatter."""
    sign = "−" if x < 0 else ""
    x = abs(x)
    for scale, suffix in ((1e15, "Q"), (1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if x >= scale:
            return f"{sign}${x / scale:g}{suffix}"
    return f"{sign}${x:g}"


def money3(x: float, _=None) -> str:
    """Like money(), rounded to 3 significant figures ($2.1M, not $2.09715M)."""
    return money(float(f"{x:.3g}"))


def count(x: float, _=None) -> str:
    """Plain count with a k/M/B suffix, e.g. 1e9 -> '1B'."""
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= scale:
            return f"{x / scale:g}{suffix}"
    return f"{x:g}"


def words(m: float) -> str:
    """4.35e17 -> '435 quadrillion'."""
    for scale, name in ((1e15, "quadrillion"), (1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if m >= scale:
            return f"{m / scale:.3g} {name}"
    return f"{m:,.0f}"


def percent(v: float, _=None) -> str:
    return f"{v:.0%}"


def dollars(v: float, _=None) -> str:
    return f"${v:.0f}"
