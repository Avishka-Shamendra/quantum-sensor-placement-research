"""Shared publication style for minimum-error figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


def configure_plot_style() -> None:
    """Use 15-point Times New Roman text in every figure."""
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.size": 15,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "figure.labelsize": 15,
            "figure.titlesize": 15,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
