#!/usr/bin/env python
"""
Generate a comparison figure for the completed model runs.

This uses the documented final metrics from the completed training runs and
creates a compact report-friendly visual summary.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "MatplotlibConfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_PATH = Path("data/manifests/training_outcomes_comparison.png")


RUNS = [
    {
        "label": "PKU\n10ep",
        "precision": 0.12047,
        "recall": 0.04990,
        "map50": 0.01893,
        "map5095": 0.00717,
        "color": "#9aa5b1",
    },
    {
        "label": "PKU\n50ep",
        "precision": 0.12634,
        "recall": 0.06582,
        "map50": 0.04391,
        "map5095": 0.01627,
        "color": "#7f8c8d",
    },
    {
        "label": "DeepPCB\nBaseline",
        "precision": 0.91156,
        "recall": 0.91519,
        "map50": 0.95582,
        "map5095": 0.71398,
        "color": "#2b8a3e",
    },
    {
        "label": "PCB v1i\nBaseline",
        "precision": 0.97831,
        "recall": 0.97537,
        "map50": 0.98915,
        "map5095": 0.57440,
        "color": "#1f77b4",
    },
    {
        "label": "DeepPCB ->\nPCB v1i Transfer",
        "precision": 0.97514,
        "recall": 0.98006,
        "map50": 0.98884,
        "map5095": 0.56531,
        "color": "#d97706",
    },
]


METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("map50", "mAP50"),
    ("map5095", "mAP50-95"),
]


def add_value_labels(ax, bars) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + (bar.get_width() / 2),
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    labels = [run["label"] for run in RUNS]
    colors = [run["color"] for run in RUNS]
    x_positions = range(len(RUNS))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (metric_key, metric_title) in zip(axes, METRICS):
        values = [run[metric_key] for run in RUNS]
        bars = ax.bar(x_positions, values, color=colors, edgecolor="#2f2f2f", linewidth=0.6)
        ax.set_title(metric_title, fontsize=13, weight="bold")
        ax.set_ylim(0, 1.08)
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(labels, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        add_value_labels(ax, bars)

    fig.suptitle("Comparison of Training Outcomes Across Completed Model Runs", fontsize=16, weight="bold")
    fig.text(
        0.5,
        0.02,
        "Metrics are taken from the completed run summaries. Higher values indicate stronger detection performance.",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"FIGURE={OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
