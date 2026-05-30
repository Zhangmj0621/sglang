from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from plot_effective_request_completion_time import configure_osdi_style


DEFAULT_TURNS = list(range(20))
DEFAULT_BASELINE = [
    0.0000,
    0.9385,
    0.8616,
    0.5506,
    0.3414,
    0.2599,
    0.2502,
    0.1774,
    0.3552,
    0.4896,
    0.8699,
    0.9652,
    0.9667,
    0.9680,
    0.9692,
    0.9704,
    0.9714,
    0.9724,
    0.9733,
    0.9742,
]
DEFAULT_PROACTIVE = [
    0.0000,
    0.9385,
    0.9428,
    0.9466,
    0.9500,
    0.9529,
    0.9555,
    0.9579,
    0.9600,
    0.9619,
    0.9636,
    0.9652,
    0.9667,
    0.9680,
    0.9692,
    0.9704,
    0.9714,
    0.9724,
    0.9733,
    0.9742,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot an OSDI-style per-turn cache hit comparison for the bsz=128 "
            "benchmark."
        )
    )
    parser.add_argument(
        "--turns",
        type=int,
        nargs="+",
        default=DEFAULT_TURNS,
        help="Turn IDs on the x-axis.",
    )
    parser.add_argument(
        "--baseline-values",
        type=float,
        nargs="+",
        default=DEFAULT_BASELINE,
        help="Per-turn cache hit rates for baseline.",
    )
    parser.add_argument(
        "--proactive-values",
        type=float,
        nargs="+",
        default=DEFAULT_PROACTIVE,
        help="Per-turn cache hit rates for the proactive KVCache manager.",
    )
    parser.add_argument(
        "--highlight-start",
        type=int,
        default=3,
        help="Start of the degraded middle-turn region to highlight.",
    )
    parser.add_argument(
        "--highlight-end",
        type=int,
        default=9,
        help="End of the degraded middle-turn region to highlight.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/hicache/figures/bsz128_cache_hit_by_turn"),
        help="Output path stem. The script writes both .pdf and .png files.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional figure title. Defaults to no title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG export DPI.",
    )
    return parser.parse_args()


def validate_lengths(
    turns: list[int], baseline_values: list[float], proactive_values: list[float]
) -> None:
    expected = len(turns)
    if len(baseline_values) != expected or len(proactive_values) != expected:
        raise ValueError(
            "The number of turns, baseline values, and proactive values must match."
        )


def plot_chart(
    turns: list[int],
    baseline_values: list[float],
    proactive_values: list[float],
    output_stem: Path,
    highlight_start: int,
    highlight_end: int,
    title: str,
    dpi: int,
) -> list[Path]:
    configure_osdi_style()

    turns_arr = np.asarray(turns, dtype=int)
    baseline = np.asarray(baseline_values, dtype=float)
    proactive = np.asarray(proactive_values, dtype=float)

    fig, ax = plt.subplots()

    if highlight_start <= highlight_end:
        ax.axvspan(
            highlight_start - 0.5,
            highlight_end + 0.5,
            color="#E5E7EB",
            alpha=0.55,
            zorder=0,
        )
        ax.text(
            (highlight_start + highlight_end) / 2,
            1.02,
            "Middle turns",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#4B5563",
        )

    ax.fill_between(
        turns_arr,
        baseline,
        proactive,
        where=proactive >= baseline,
        color="#4C78A8",
        alpha=0.12,
        zorder=1,
    )

    ax.plot(
        turns_arr,
        baseline,
        label="Baseline",
        color="#6B7280",
        linewidth=2.1,
        linestyle="--",
        marker="o",
        markersize=5.8,
        markerfacecolor="white",
        markeredgewidth=1.4,
        zorder=3,
    )
    ax.plot(
        turns_arr,
        proactive,
        label="Proactive KVCache Manager",
        color="#4C78A8",
        linewidth=2.4,
        marker="s",
        markersize=5.2,
        zorder=4,
    )

    valid_mask = turns_arr >= 1
    valid_indices = np.where(valid_mask)[0]
    min_idx = valid_indices[int(np.argmin(baseline[valid_mask]))]
    min_turn = turns_arr[min_idx]
    min_baseline = baseline[min_idx]
    ax.scatter(
        [min_turn],
        [min_baseline],
        color="#B91C1C",
        s=34,
        zorder=5,
    )
    ax.annotate(
        f"Baseline trough\nTurn {min_turn}: {min_baseline * 100:.1f}%",
        xy=(min_turn, min_baseline),
        xytext=(min_turn + 1.2, min_baseline + 0.20),
        textcoords="data",
        fontsize=10,
        color="#7F1D1D",
        arrowprops={
            "arrowstyle": "->",
            "lw": 1.0,
            "color": "#7F1D1D",
        },
    )

    ax.set_xlabel("Turn ID")
    ax.set_ylabel("Per-turn Cache Hit Rate")
    ax.set_xticks(turns_arr)
    ax.set_xlim(turns_arr.min() - 0.4, turns_arr.max() + 0.4)
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.legend(loc="lower right")

    if title:
        ax.set_title(title, pad=8)

    ax.text(
        0.01,
        0.98,
        "Batch Size = 128",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#4B5563",
    )
    ax.text(
        0.99,
        0.98,
        "Higher is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#4B5563",
    )

    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    return [pdf_path, png_path]


def main() -> None:
    args = parse_args()
    validate_lengths(args.turns, args.baseline_values, args.proactive_values)
    saved_paths = plot_chart(
        turns=args.turns,
        baseline_values=args.baseline_values,
        proactive_values=args.proactive_values,
        output_stem=args.output,
        highlight_start=args.highlight_start,
        highlight_end=args.highlight_end,
        title=args.title,
        dpi=args.dpi,
    )
    print("Saved figure(s):")
    for path in saved_paths:
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
