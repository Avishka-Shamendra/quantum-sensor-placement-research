"""Shared style for minimum-error figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

METHOD_STYLES = {
    "Optimal": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "Exhaustive": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "Greedy": {"color": "#E69F00", "linestyle": "--", "marker": "s"},
    "Heuristic": {"color": "#D55E00", "linestyle": "-.", "marker": "D"},
    "Uniform": {"color": "#6F6F6F", "linestyle": ":", "marker": "^"},
    "Random": {"color": "#009E73", "linestyle": (0, (5, 2)), "marker": "x"},
}
METHOD_COLORS = {
    method: style["color"] for method, style in METHOD_STYLES.items()
}
METHOD_LINESTYLES = {
    method: style["linestyle"] for method, style in METHOD_STYLES.items()
}
METHOD_MARKERS = {
    method: style["marker"] for method, style in METHOD_STYLES.items()
}
BEAM_WIDTH_COLORS = ("#0072B2", "#009E73", "#D55E00", "#E69F00", "#6F6F6F")
FIELD_FAMILY_COLORS = {
    "gaussian": "#0072B2",
    "sinusoidal": "#009E73",
    "quadratic": "#D55E00",
}
REFERENCE_LINE_COLOR = "#202020"


def configure_plot_style() -> None:
    """Use consistent typography and axes in every figure."""
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.labelsize": 12,
            "figure.titlesize": 12,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
