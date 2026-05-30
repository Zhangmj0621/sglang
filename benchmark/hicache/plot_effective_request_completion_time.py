from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BATCH_SIZES = [64, 96, 128]
DEFAULT_BASELINE = [295.08, 451.30, 1311.54]
DEFAULT_PROACTIVE = [269.26, 369.56, 474.72]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot an OSDI-style grouped bar chart for effective request "
            "completion time."
        )
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help=(
            "Batch sizes shown on the x-axis. Defaults to the three values in "
            "the pasted benchmark logs: 64 96 128."
        ),
    )
    parser.add_argument(
        "--baseline-values",
        type=float,
        nargs="+",
        default=DEFAULT_BASELINE,
        help="Baseline effective completion times in seconds.",
    )
    parser.add_argument(
        "--proactive-values",
        type=float,
        nargs="+",
        default=DEFAULT_PROACTIVE,
        help="Proactive KVCache manager effective completion times in seconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/hicache/figures/effective_request_completion_time"),
        help="Output path stem. The script writes both .pdf and .png files.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional figure title. OSDI-style figures usually omit titles.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG export DPI.",
    )
    return parser.parse_args()


def configure_osdi_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (7.2, 4.2),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "grid.color": "#D0D7DE",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "hatch.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def validate_lengths(
    batch_sizes: list[int], baseline_values: list[float], proactive_values: list[float]
) -> None:
    expected = len(batch_sizes)
    if len(baseline_values) != expected or len(proactive_values) != expected:
        raise ValueError(
            "The number of batch sizes, baseline values, and proactive values "
            "must match."
        )


def annotate_bars(ax: plt.Axes, bars: list[plt.Rectangle], ymax: float) -> None:
    offset = ymax * 0.018
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1F2937",
        )


def annotate_improvements(
    ax: plt.Axes,
    x: np.ndarray,
    baseline_values: np.ndarray,
    proactive_values: np.ndarray,
    ymax: float,
) -> None:
    for idx, (base, proactive) in enumerate(zip(baseline_values, proactive_values)):
        reduction = (base - proactive) / base * 100.0
        ax.text(
            x[idx],
            max(base, proactive) + ymax * 0.065,
            f"-{reduction:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#1D4ED8",
        )


def plot_chart(
    batch_sizes: list[int],
    baseline_values: list[float],
    proactive_values: list[float],
    output_stem: Path,
    title: str,
    dpi: int,
) -> list[Path]:
    configure_osdi_style()

    x = np.arange(len(batch_sizes), dtype=float)
    width = 0.34

    baseline = np.asarray(baseline_values, dtype=float)
    proactive = np.asarray(proactive_values, dtype=float)
    ymax = max(float(np.max(baseline)), float(np.max(proactive)))

    fig, ax = plt.subplots()

    baseline_bars = ax.bar(
        x - width / 2,
        baseline,
        width,
        label="Baseline",
        color="#B8C0CC",
        edgecolor="#4B5563",
        linewidth=1.0,
        hatch="//",
        zorder=3,
    )
    proactive_bars = ax.bar(
        x + width / 2,
        proactive,
        width,
        label="Proactive KVCache Manager",
        color="#4C78A8",
        edgecolor="#1F2937",
        linewidth=1.0,
        zorder=3,
    )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Effective Request Completion Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in batch_sizes])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.legend(loc="upper left", ncol=2)

    if title:
        ax.set_title(title, pad=8)

    ax.text(
        0.99,
        0.98,
        "Lower is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#4B5563",
    )

    ax.set_ylim(0, ymax * 1.22)

    annotate_bars(ax, list(baseline_bars), ymax)
    annotate_bars(ax, list(proactive_bars), ymax)
    annotate_improvements(ax, x, baseline, proactive, ymax)

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
    validate_lengths(args.batch_sizes, args.baseline_values, args.proactive_values)
    saved_paths = plot_chart(
        batch_sizes=args.batch_sizes,
        baseline_values=args.baseline_values,
        proactive_values=args.proactive_values,
        output_stem=args.output,
        title=args.title,
        dpi=args.dpi,
    )
    print("Saved figure(s):")
    for path in saved_paths:
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
