"""Throughput-interactivity frontier chart from one or two sweep CSVs.

Each CSV row is one concurrency level. The chart puts speed per user
(1000 / tpot_ms, tokens/s) on x and throughput per GPU on y, joins the
Pareto-optimal points, and shades the region below an interactivity floor.
The operating point is the highest-throughput level at or above the floor; it
is ringed on the chart and spelled out in the legend.

Run:
    python plot_frontier.py results/sweep_0.5b_cuda-graphs.csv --label "CUDA graphs" \
        --overlay results/sweep_0.5b_eager.csv --overlay-label "eager" \
        --out results/frontier_0.5b_knob.png
"""

import argparse
import csv
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # write files only; the demo opens the PNG itself
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

# Categorical slots 1 and 2 of the dataviz reference palette; text stays in ink.
SERIES_COLORS = ("#2a78d6", "#eb6834")
INK, MUTED, GRID, FLOOR_FILL = "#0b0b0b", "#52514e", "#e4e3df", "#f1f0ec"
Y_LABELS = {
    "output_tok_s_per_gpu": "output throughput per GPU (tokens/s)",
    "total_tok_s_per_gpu": "total throughput per GPU (tokens/s)",
}


@dataclass(frozen=True)
class Point:
    """One concurrency level: speed per user (x) and throughput per GPU (y)."""

    concurrency: int
    speed: float
    throughput: float


@dataclass(frozen=True)
class Series:
    """One sweep, drawn in one color."""

    label: str
    color: str
    points: tuple[Point, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", help="Sweep CSV from sweep.py.")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--overlay", help="A second sweep CSV, such as the knob turned off.")
    parser.add_argument("--overlay-label", default="variant")
    parser.add_argument("--y-column", default="output_tok_s_per_gpu",
                        help="Throughput column; any column ending in tok_s_per_gpu.")
    parser.add_argument("--floor", type=float, default=30.0,
                        help="Slowest speed per user to accept, in tokens/s.")
    parser.add_argument("--title", default="Throughput per GPU against speed per user")
    parser.add_argument("--out", default="results/frontier.png")
    return parser.parse_args()


def load_series(path: str, label: str, color: str, y_column: str) -> Series:
    """Read a sweep CSV into points, converting TPOT (ms) to tokens/s per user."""
    with open(path, newline="") as f:
        points = tuple(
            Point(int(row["concurrency"]), 1000.0 / float(row["tpot_ms"]), float(row[y_column]))
            for row in csv.DictReader(f)
        )
    return Series(label, color, points)


def pareto(points: tuple[Point, ...]) -> list[Point]:
    """Points that no other point beats on both speed and throughput, by speed."""
    frontier, best = [], float("-inf")
    for p in sorted(points, key=lambda p: -p.speed):
        if p.throughput > best:
            frontier.append(p)
            best = p.throughput
    return frontier


def operating_point(series: Series, floor: float) -> Point | None:
    """Highest-throughput point whose speed per user meets the floor."""
    eligible = [p for p in series.points if p.speed >= floor]
    return max(eligible, key=lambda p: p.throughput, default=None)


def label_concurrency(ax, points: tuple[Point, ...]) -> None:
    """Write each point's concurrency beside it, below-left when it would crowd the last."""
    x_span = max(p.speed for p in points)
    y_span = max(p.throughput for p in points)
    previous = None
    for p in sorted(points, key=lambda p: p.concurrency):
        crowded = (previous is not None
                   and abs(p.speed - previous.speed) < 0.04 * x_span
                   and abs(p.throughput - previous.throughput) < 0.04 * y_span)
        offset, align = ((-6, -12), "right") if crowded else ((6, 4), "left")
        ax.annotate(f"{p.concurrency}", (p.speed, p.throughput), textcoords="offset points",
                    xytext=offset, ha=align, fontsize=8, color=MUTED)
        previous = p


def describe(series: Series, op: Point | None) -> str:
    """Legend entry: the series label and its operating point, if it has one."""
    if op is None:
        return f"{series.label}: no level meets the floor"
    return (f"{series.label}: {op.concurrency} users, {op.throughput:,.0f} tok/s per GPU "
            f"at {op.speed:.0f} tok/s each")


def draw_series(ax, series: Series, floor: float) -> None:
    """Frontier line, every measured point, and the ringed operating point."""
    front = pareto(series.points)
    op = operating_point(series, floor)
    ax.plot([p.speed for p in front], [p.throughput for p in front], color=series.color,
            linewidth=2, label=describe(series, op), zorder=3)
    ax.scatter([p.speed for p in series.points], [p.throughput for p in series.points],
               s=36, color=series.color, edgecolor="white", linewidth=1.5, zorder=4)
    label_concurrency(ax, series.points)
    if op is not None:
        ax.scatter([op.speed], [op.throughput], s=220, facecolor="none",
                   edgecolor=series.color, linewidth=2, zorder=5)


def draw_floor(ax, floor: float) -> None:
    """Shade speeds below the floor and draw it as a dashed line; the legend names it."""
    ax.axvspan(0, floor, color=FLOOR_FILL, zorder=0)
    ax.axvline(floor, color=MUTED, linestyle="--", linewidth=1, zorder=1)


def style_axes(ax, y_column: str, title: str, y_max: float) -> None:
    """Zero-based axes, recessive grid, and plain-language labels."""
    ax.set_xlim(left=0)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("speed per user (tokens/s) = 1000 / TPOT", color=INK)
    ax.set_ylabel(Y_LABELS.get(y_column, "throughput per GPU (tokens/s)"), color=INK)
    ax.set_title(title, color=INK, loc="left")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    args = parse_args()
    series = [load_series(args.csv, args.label, SERIES_COLORS[0], args.y_column)]
    if args.overlay:
        series.append(load_series(args.overlay, args.overlay_label, SERIES_COLORS[1],
                                  args.y_column))
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    for s in series:
        draw_series(ax, s, args.floor)
    # A quarter of headroom keeps the legend above the data.
    top = max(p.throughput for s in series for p in s.points) * 1.25
    style_axes(ax, args.y_column, args.title, top)
    draw_floor(ax, args.floor)
    ax.legend(frameon=False, loc="upper right", fontsize=9, title_fontsize=9,
              title=f"dashed: floor of {args.floor:g} tok/s per user; ringed: best point above it",
              alignment="left")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
