"""Evaluate perfect-pairwise datasets stored below ``inputs/``.

Examples from the repository root:

    python3 perfect-pairwise/eval.py gaussian-small --mode both
    python3 perfect-pairwise/eval.py fourier-large --mode greedy

``--mode auto`` (the default) runs both algorithms for cases with ``K <= 10``
and only the greedy algorithm for larger cases.
"""

import argparse
import csv
import json
import sys
from html import escape
from math import ceil, log
from pathlib import Path
from statistics import fmean, median
from time import perf_counter


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
ROOT_DIRECTORY = SCRIPT_DIRECTORY.parent
INPUT_ROOT = ROOT_DIRECTORY / "inputs"
OUTPUT_ROOT = SCRIPT_DIRECTORY / "output"

if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

from util import compute_nijm, load_rijm_csv  # noqa: E402
from greedy import greedy_perfect_pairwise_sensor_placement  # noqa: E402
from optimal import optimal_perfect_pairwise_sensor_placement  # noqa: E402


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate perfect-pairwise CSV datasets."
    )
    parser.add_argument(
        "dataset",
        help="Dataset directory under inputs/, or 'all'.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "both", "optimal", "greedy"),
        default="auto",
        help="Algorithms to run. Auto uses both for K <= 10 and greedy for K > 10.",
    )
    parser.add_argument(
        "--repeats",
        type=_positive_integer,
        default=1,
        help=(
            "Number of deterministic solver runs per case when timing; "
            "reported runtimes are mean seconds per run (default: 1)."
        ),
    )
    return parser.parse_args()


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _dataset_directories(dataset_name: str) -> list[Path]:
    if not INPUT_ROOT.is_dir():
        raise ValueError(f"Inputs directory does not exist: {INPUT_ROOT}")

    available = sorted(
        path
        for path in INPUT_ROOT.iterdir()
        if path.is_dir() and any(path.glob("*.csv"))
    )
    if dataset_name == "all":
        if not available:
            raise ValueError(f"No dataset directories found in: {INPUT_ROOT}")
        return available

    selected = INPUT_ROOT / dataset_name
    if not selected.is_dir():
        names = ", ".join(path.name for path in available) or "none"
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {names}.")
    return [selected]


def _mode_for_case(requested_mode: str, F: int) -> str:
    if F > 10 and requested_mode in ("both", "optimal"):
        raise ValueError(
            f"K={F} exceeds the exact-solver limit (K <= 10). "
            "Use --mode greedy or --mode auto."
        )
    if requested_mode != "auto":
        return requested_mode
    return "both" if F <= 10 else "greedy"


def _pair_keys_for_json(
    values: dict[tuple[int, int], list[int | float]],
) -> dict[str, list[int | float]]:
    return {f"{i},{j}": value for (i, j), value in values.items()}


def _run_repeatedly(
    solver: object,
    F: int,
    M: int,
    Nijm: dict[tuple[int, int], list[int]],
    repetitions: int,
) -> tuple[tuple[int, list[int], list[list[int | None]]], float]:
    """Run one deterministic solver repeatedly and return mean time per run."""
    if not callable(solver):
        raise TypeError("solver must be callable")

    result: tuple[int, list[int], list[list[int | None]]] | None = None
    started = perf_counter()
    for _ in range(repetitions):
        candidate = solver(F, M, Nijm)
        if result is None:
            result = candidate
        elif candidate != result:
            raise RuntimeError("A deterministic solver returned inconsistent results.")
    assert result is not None
    return result, (perf_counter() - started) / repetitions


def _build_record(
    case_name: str,
    F: int,
    M: int,
    rijm: dict[tuple[int, int], list[float]],
    Nijm: dict[tuple[int, int], list[int]],
    mode: str,
    repetitions: int,
) -> dict[str, object]:
    """Run the requested algorithms and prepare a JSON-ready record."""
    record: dict[str, object] = {
        "case": case_name,
        "F": F,
        "M": M,
        "pair_count": F * (F - 1) // 2,
        "mode": mode,
        "rijm": _pair_keys_for_json(rijm),
        "Nijm": _pair_keys_for_json(Nijm),
        "optimal": None,
        "greedy": None,
        "greedy_approximation_bound": None,
        "timing_seconds": {"optimal": None, "greedy": None},
        "timing_repetitions": repetitions,
        "metrics": {
            "greedy_minus_optimal": None,
            "greedy_to_optimal_ratio": None,
            "bound_utilization": None,
        },
    }

    optimal_total: int | None = None
    if mode in ("both", "optimal"):
        (optimal_total, allocation, pair_locations), optimal_time = _run_repeatedly(
            optimal_perfect_pairwise_sensor_placement, F, M, Nijm, repetitions
        )
        record["optimal"] = {
            "Nmin": optimal_total,
            "allocation": allocation,
            "pair_locations": pair_locations,
        }
        timing = record["timing_seconds"]
        assert isinstance(timing, dict)
        timing["optimal"] = optimal_time

    greedy_total: int | None = None
    if mode in ("both", "greedy"):
        (greedy_total, allocation, pair_locations), greedy_time = _run_repeatedly(
            greedy_perfect_pairwise_sensor_placement, F, M, Nijm, repetitions
        )
        record["greedy"] = {
            "Nmin": greedy_total,
            "allocation": allocation,
            "pair_locations": pair_locations,
        }
        timing = record["timing_seconds"]
        assert isinstance(timing, dict)
        timing["greedy"] = greedy_time

    # The paper's approximation bound requires the exact N_optimal value.
    if mode == "both" and optimal_total is not None:
        record["greedy_approximation_bound"] = (
            log(int(record["pair_count"])) + 1
        ) * optimal_total

    if optimal_total is not None and greedy_total is not None:
        metrics = record["metrics"]
        assert isinstance(metrics, dict)
        metrics["greedy_minus_optimal"] = greedy_total - optimal_total
        metrics["greedy_to_optimal_ratio"] = greedy_total / optimal_total
        bound = record["greedy_approximation_bound"]
        if bound is not None:
            metrics["bound_utilization"] = greedy_total / float(bound)

    return record


def _optional_nmin(record: dict[str, object], algorithm: str) -> int | None:
    result = record[algorithm]
    if result is None:
        return None
    assert isinstance(result, dict)
    return int(result["Nmin"])


def _write_case_output(
    dataset_output: Path, record: dict[str, object], family: str
) -> dict[str, int | float | str | None]:
    case_name = str(record["case"])
    case_output = dataset_output / case_name
    case_output.mkdir(parents=True, exist_ok=True)

    with (case_output / "results.json").open("w", encoding="utf-8") as output_file:
        json.dump(record, output_file, indent=2)
        output_file.write("\n")

    row = {
        "case": case_name,
        "family": family,
        "F": int(record["F"]),
        "M": int(record["M"]),
        "mode": str(record["mode"]),
        "Nmin_optimal": _optional_nmin(record, "optimal"),
        "Nmin_greedy": _optional_nmin(record, "greedy"),
        "greedy_approximation_bound": record["greedy_approximation_bound"],
        "timing_repetitions": int(record["timing_repetitions"]),
    }
    metrics = record["metrics"]
    timing = record["timing_seconds"]
    assert isinstance(metrics, dict) and isinstance(timing, dict)
    row["greedy_minus_optimal"] = metrics["greedy_minus_optimal"]
    row["greedy_to_optimal_ratio"] = metrics["greedy_to_optimal_ratio"]
    row["bound_utilization"] = metrics["bound_utilization"]
    row["optimal_seconds"] = timing["optimal"]
    row["greedy_seconds"] = timing["greedy"]

    _write_csv(case_output / "comparison.csv", [row])
    return row


def _family_from_case_name(case_name: str) -> str:
    """Best-effort fallback for CSV files without a dataset manifest."""
    prefix = case_name.split("_k", maxsplit=1)[0]
    return prefix.rsplit("_", maxsplit=1)[0]


def _case_families_from_manifest(dataset_directory: Path) -> dict[str, str]:
    """Read optional generator metadata mapping each CSV name to its family."""
    manifest_path = dataset_directory / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(manifest, dict):
        return {}

    default_family = manifest.get("field_family")
    families: dict[str, str] = {}
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        return families
    for case in cases:
        if not isinstance(case, dict):
            continue
        filename = case.get("file")
        family = case.get("field_family", default_family)
        if isinstance(filename, str) and isinstance(family, str):
            families[Path(filename).stem] = family
    return families


def _write_csv(
    path: Path, rows: list[dict[str, int | float | str | None]]
) -> None:
    columns = [
        "case",
        "family",
        "F",
        "M",
        "mode",
        "Nmin_optimal",
        "Nmin_greedy",
        "greedy_approximation_bound",
        "timing_repetitions",
        "greedy_minus_optimal",
        "greedy_to_optimal_ratio",
        "bound_utilization",
        "optimal_seconds",
        "greedy_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=columns, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _print_table(dataset_name: str, rows: list[dict[str, int | float | str | None]]) -> None:
    headers = [
        "Case", "Family", "K", "M", "Mode", "Nmin optimal",
        "Nmin greedy", "Gap", "Ratio", "Bound use", "Opt/run (s)",
        "Greedy/run (s)", "Total/run (s)",
    ]

    def display(value: int | float | str | None, precision: int = 0) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        return str(value)

    table_rows = [
        [
            display(row["case"]),
            display(row["family"]),
            display(row["F"]),
            display(row["M"]),
            display(row["mode"]),
            display(row["Nmin_optimal"]),
            display(row["Nmin_greedy"]),
            display(row["greedy_minus_optimal"]),
            display(row["greedy_to_optimal_ratio"], precision=3),
            display(row["bound_utilization"], precision=3),
            display(row["optimal_seconds"], precision=6),
            display(row["greedy_seconds"], precision=6),
            display(
                (row["optimal_seconds"] or 0) + (row["greedy_seconds"] or 0),
                precision=6,
            ),
        ]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in table_rows))
        for index, header in enumerate(headers)
    ]

    def format_row(row: list[str]) -> str:
        return " | ".join(value.rjust(width) for value, width in zip(row, widths))

    print(f"\nDataset: {dataset_name}")
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(format_row(row))


def _family_summary_rows(
    rows: list[dict[str, int | float | str | None]],
) -> list[dict[str, int | float | str | None]]:
    """Create descriptive, per-family summaries for paper-ready reporting."""
    grouped: dict[str, list[dict[str, int | float | str | None]]] = {}
    for row in rows:
        family = str(row["family"])
        grouped.setdefault(family, []).append(row)

    summaries: list[dict[str, int | float | str | None]] = []
    for family, family_rows in grouped.items():
        exact_rows = [row for row in family_rows if row["Nmin_optimal"] is not None]

        def average(key: str) -> float | None:
            values = [float(row[key]) for row in exact_rows if row[key] is not None]
            return fmean(values) if values else None

        ratios = [
            float(row["greedy_to_optimal_ratio"])
            for row in exact_rows
            if row["greedy_to_optimal_ratio"] is not None
        ]
        optimal_times = [
            float(row["optimal_seconds"])
            for row in family_rows
            if row["optimal_seconds"] is not None
        ]
        greedy_times = [
            float(row["greedy_seconds"])
            for row in family_rows
            if row["greedy_seconds"] is not None
        ]
        summaries.append({
            "family": family,
            "case_count": len(family_rows),
            "exact_case_count": len(exact_rows),
            "mean_Nmin_optimal": average("Nmin_optimal"),
            "mean_Nmin_greedy": average("Nmin_greedy"),
            "mean_greedy_gap": average("greedy_minus_optimal"),
            "mean_greedy_to_optimal_ratio": fmean(ratios) if ratios else None,
            "median_greedy_to_optimal_ratio": median(ratios) if ratios else None,
            "max_greedy_to_optimal_ratio": max(ratios) if ratios else None,
            "mean_optimal_seconds": fmean(optimal_times) if optimal_times else None,
            "mean_greedy_seconds": fmean(greedy_times) if greedy_times else None,
        })
    return summaries


def _write_family_summary(
    path: Path, rows: list[dict[str, int | float | str | None]]
) -> None:
    columns = [
        "family",
        "case_count",
        "exact_case_count",
        "mean_Nmin_optimal",
        "mean_Nmin_greedy",
        "mean_greedy_gap",
        "mean_greedy_to_optimal_ratio",
        "median_greedy_to_optimal_ratio",
        "max_greedy_to_optimal_ratio",
        "mean_optimal_seconds",
        "mean_greedy_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=columns, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(_family_summary_rows(rows))


_FAMILY_STYLES = {
    "gaussian": ("#1a73e8", "Gaussian"),
    "sine": ("#d93025", "Sine"),
    "fourier": ("#188038", "Fourier"),
    "polynomial": ("#9334e6", "Polynomial"),
}


def _svg_position(
    value: float, lower: float, upper: float, start: float, length: float
) -> float:
    """Map a value from an axis interval to an SVG interval."""
    if upper == lower:
        return start + length / 2
    return start + (value - lower) * length / (upper - lower)


def _integer_ticks(upper: float, maximum_count: int = 8) -> list[tuple[float, str]]:
    """Return a readable set of integer tick labels for a non-negative axis."""
    integer_upper = int(ceil(upper))
    step = max(1, int(ceil(integer_upper / maximum_count)))
    values = list(range(0, integer_upper + 1, step))
    if values[-1] != integer_upper:
        values.append(integer_upper)
    return [(float(value), str(value)) for value in values]


def _ratio_ticks(upper: float) -> list[tuple[float, str]]:
    """Return concise, readable ticks for an approximation-ratio axis."""
    step = 0.1 if upper <= 1.6 else 0.2 if upper <= 2.2 else 0.5
    tick_count = int(round((upper - 1.0) / step))
    return [
        (round(1.0 + index * step, 2), f"{1.0 + index * step:.1f}")
        for index in range(tick_count + 1)
    ]


def _controlled_k_sweep(
    rows: list[dict[str, int | float | str | None]],
) -> tuple[list[dict[str, int | float | str | None]], int]:
    """Select the fixed-M slice containing the most distinct K values."""
    candidates = sorted({int(row["M"]) for row in rows})
    fixed_M = max(
        candidates,
        key=lambda M: (
            len({int(row["F"]) for row in rows if int(row["M"]) == M}),
            sum(1 for row in rows if int(row["M"]) == M),
            -M,
        ),
    )
    return [row for row in rows if int(row["M"]) == fixed_M], fixed_M


def _controlled_m_sweep(
    rows: list[dict[str, int | float | str | None]],
) -> tuple[list[dict[str, int | float | str | None]], int]:
    """Select the fixed-K slice containing the most distinct M values."""
    candidates = sorted({int(row["F"]) for row in rows})
    fixed_K = max(
        candidates,
        key=lambda K: (
            len({int(row["M"]) for row in rows if int(row["F"]) == K}),
            sum(1 for row in rows if int(row["F"]) == K),
            -K,
        ),
    )
    return [row for row in rows if int(row["F"]) == fixed_K], fixed_K


def _present_families(
    rows: list[dict[str, int | float | str | None]],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(row["family"])
            for row in rows
            if str(row["family"]) in _FAMILY_STYLES
        )
    )


def _append_axes(
    lines: list[str],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    x_lower: float,
    x_upper: float,
    y_lower: float,
    y_upper: float,
    x_ticks: list[tuple[float, str]],
    y_ticks: list[tuple[float, str]],
    x_label: str,
    y_label: str,
) -> None:
    """Append labelled linear axes for one compact SVG panel."""
    bottom = top + height
    right = left + width
    for value, label in y_ticks:
        y = _svg_position(value, y_lower, y_upper, bottom, -height)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>')
        lines.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{escape(label)}</text>')
    for value, label in x_ticks:
        x = _svg_position(value, x_lower, x_upper, left, width)
        lines.append(f'<line class="axis" x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom + 5}"/>')
        lines.append(f'<text class="tick" x="{x:.2f}" y="{bottom + 20}" text-anchor="middle">{escape(label)}</text>')
    lines.extend([
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
        f'<text class="label" x="{left + width / 2}" y="{bottom + 46}" text-anchor="middle">{escape(x_label)}</text>',
        f'<text class="label" x="{left - 48}" y="{top + height / 2}" transform="rotate(-90 {left - 48} {top + height / 2})" text-anchor="middle">{escape(y_label)}</text>',
    ])


def _append_bound_axis(
    lines: list[str],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    lower: float,
    upper: float,
) -> None:
    """Append a purple right-hand scale for the theorem bound B(K)."""
    right = left + width
    bottom = top + height
    ticks = range(1, int(upper) + 1)
    lines.append(
        f'<line class="bound-axis" x1="{right}" y1="{top}" '
        f'x2="{right}" y2="{bottom}"/>'
    )
    for value in ticks:
        y = _svg_position(float(value), lower, upper, bottom, -height)
        lines.append(
            f'<line class="bound-axis" x1="{right}" y1="{y:.2f}" '
            f'x2="{right + 5}" y2="{y:.2f}"/>'
        )
        lines.append(
            f'<text class="bound-tick" x="{right + 9}" y="{y + 4:.2f}">{value}</text>'
        )
    label_x = right + 47
    label_y = top + height / 2
    lines.append(
        f'<text class="bound-label" x="{label_x}" y="{label_y}" '
        f'transform="rotate(90 {label_x} {label_y})" text-anchor="middle">'
        "B(K) theorem bound</text>"
    )


def _append_marker(
    lines: list[str], x: float, y: float, family: str, size: float = 5
) -> None:
    """Append a colour-and-shape marker so figures remain readable in print."""
    color, _ = _FAMILY_STYLES.get(family, ("#5f6368", family.replace("_", " ")))
    if family == "gaussian":
        lines.append(f'<rect x="{x - size:.2f}" y="{y - size:.2f}" width="{2 * size}" height="{2 * size}" fill="{color}" opacity="0.88"/>')
    elif family == "sine":
        points = f"{x:.2f},{y - size:.2f} {x + size:.2f},{y + size:.2f} {x - size:.2f},{y + size:.2f}"
        lines.append(f'<polygon points="{points}" fill="{color}" opacity="0.88"/>')
    elif family == "polynomial":
        points = f"{x:.2f},{y - size:.2f} {x + size:.2f},{y:.2f} {x:.2f},{y + size:.2f} {x - size:.2f},{y:.2f}"
        lines.append(f'<polygon points="{points}" fill="{color}" opacity="0.88"/>')
    else:
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size}" fill="{color}" opacity="0.88"/>')


def _append_family_legend(
    lines: list[str], x: float, y: float, families: tuple[str, ...] | None = None
) -> None:
    families = families or tuple(_FAMILY_STYLES)
    for index, family in enumerate(families):
        _, label = _FAMILY_STYLES[family]
        marker_x = x + 150 * index
        _append_marker(lines, marker_x, y - 4, family, size=5)
        lines.append(f'<text class="legend" x="{marker_x + 11}" y="{y}">{label}</text>')


def _svg_document(width: int, height: int, title: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.axis{stroke:#5f6368;stroke-width:1.2}.grid{stroke:#dadce0;stroke-width:1}.diagonal{stroke:#5f6368;stroke-width:2;stroke-dasharray:6 5}.bound{stroke:#9334e6;stroke-width:2.5;fill:none;stroke-dasharray:8 6}.bound-axis{stroke:#9334e6;stroke-width:1.2}.bound-tick{font-size:11px;fill:#9334e6}.bound-label{font-size:12px;fill:#9334e6}.tick{font-size:11px}.label{font-size:13px}.legend{font-size:12px}.panel{font-size:14px;font-weight:bold}.title{font-size:18px;font-weight:bold}.note{font-size:11px;fill:#5f6368}</style>',
        f'<text class="title" x="{width / 2}" y="25" text-anchor="middle">{escape(title)}</text>',
    ]


def _write_algorithm_quality_figure(
    dataset_output: Path,
    rows: list[dict[str, int | float | str | None]],
) -> None:
    """Write a compact two-panel exact-versus-greedy paper figure."""
    exact_rows = [row for row in rows if row["Nmin_optimal"] is not None]
    if not exact_rows:
        return

    width, height = 980, 440
    left_panel = (80.0, 80.0, 340.0, 270.0)
    right_panel = (585.0, 80.0, 340.0, 270.0)
    nmax = max(
        max(float(row["Nmin_optimal"]), float(row["Nmin_greedy"]))
        for row in exact_rows
    )
    n_upper = float(ceil(nmax + 1))
    n_ticks = _integer_ticks(n_upper)
    k_sweep_rows, fixed_M = _controlled_k_sweep(exact_rows)
    k_values = sorted({int(row["F"]) for row in k_sweep_rows})
    bound_values = [(K, log(K * (K - 1) / 2) + 1) for K in k_values]
    ratio_upper = float(max(2, ceil(max(value for _, value in bound_values))))
    ratio_ticks = [(float(value), f"{value:.0f}") for value in range(1, int(ratio_upper) + 1)]
    lines = _svg_document(width, height, "Perfect Pairwise Discrimination: Algorithm Quality")
    _append_family_legend(lines, 360, 51, _present_families(exact_rows))

    left, top, panel_width, panel_height = left_panel
    lines.append(f'<text class="panel" x="{left + panel_width / 2}" y="63" text-anchor="middle">Sensor requirement comparison</text>')
    _append_axes(
        lines, left=left, top=top, width=panel_width, height=panel_height,
        x_lower=0, x_upper=n_upper, y_lower=0, y_upper=n_upper,
        x_ticks=n_ticks, y_ticks=n_ticks,
        x_label="Nmin (optimal)", y_label="Nmin (greedy)",
    )
    diagonal_start = _svg_position(0, 0, n_upper, left, panel_width)
    diagonal_end = _svg_position(n_upper, 0, n_upper, left, panel_width)
    y_start = _svg_position(0, 0, n_upper, top + panel_height, -panel_height)
    y_end = _svg_position(n_upper, 0, n_upper, top + panel_height, -panel_height)
    lines.append(f'<line class="diagonal" x1="{diagonal_start:.2f}" y1="{y_start:.2f}" x2="{diagonal_end:.2f}" y2="{y_end:.2f}"/>')
    lines.append(f'<text class="note" x="{left + panel_width - 4}" y="{top + 15}" text-anchor="end">dotted diagonal: equal solutions</text>')
    for row in exact_rows:
        x = _svg_position(float(row["Nmin_optimal"]), 0, n_upper, left, panel_width)
        y = _svg_position(float(row["Nmin_greedy"]), 0, n_upper, top + panel_height, -panel_height)
        _append_marker(lines, x, y, str(row["family"]))

    left, top, panel_width, panel_height = right_panel
    lines.append(f'<text class="panel" x="{left + panel_width / 2}" y="63" text-anchor="middle">Greedy quality: fixed M = {fixed_M}</text>')
    _append_axes(
        lines, left=left, top=top, width=panel_width, height=panel_height,
        x_lower=min(k_values) - 0.5, x_upper=max(k_values) + 0.5,
        y_lower=1, y_upper=ratio_upper,
        x_ticks=[(float(K), str(K)) for K in k_values], y_ticks=ratio_ticks,
        x_label="Number of fields, K", y_label="Ngreedy / Noptimal",
    )
    bound_points = " ".join(
        f"{_svg_position(K, min(k_values) - 0.5, max(k_values) + 0.5, left, panel_width):.2f},{_svg_position(value, 1, ratio_upper, top + panel_height, -panel_height):.2f}"
        for K, value in bound_values
    )
    lines.append(f'<polyline class="bound" points="{bound_points}"/>')
    lines.append(f'<text class="note" x="{left + panel_width - 4}" y="{top + 15}" text-anchor="end">dotted: 1 + ln(K(K−1)/2)</text>')
    family_offsets = {"gaussian": -0.18, "sine": -0.06, "fourier": 0.06, "polynomial": 0.18}
    for row in k_sweep_rows:
        x_value = float(row["F"]) + family_offsets.get(str(row["family"]), 0.0)
        x = _svg_position(x_value, min(k_values) - 0.5, max(k_values) + 0.5, left, panel_width)
        y = _svg_position(float(row["greedy_to_optimal_ratio"]), 1, ratio_upper, top + panel_height, -panel_height)
        _append_marker(lines, x, y, str(row["family"]))

    lines.append('<text class="note" x="490" y="418" text-anchor="middle">Each marker is one generated instance; no filename labels are used.</text>')
    lines.append("</svg>")
    (dataset_output / "algorithm_quality.svg").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_cross_family_quality_figure(
    summary_output: Path,
    rows: list[dict[str, int | float | str | None]],
) -> None:
    """Write one paper-ready, four-family approximation-quality figure."""
    exact_rows = [
        row
        for row in rows
        if row["Nmin_optimal"] is not None
        and row["greedy_to_optimal_ratio"] is not None
        and str(row["family"]) in _FAMILY_STYLES
    ]
    families = tuple(
        family
        for family in _FAMILY_STYLES
        if any(str(row["family"]) == family for row in exact_rows)
    )
    if len(families) < 2:
        return

    k_values = sorted({int(row["F"]) for row in exact_rows})
    k_lower = min(k_values) - 0.5
    k_upper = max(k_values) + 0.5
    observed_maximum = max(
        float(row["greedy_to_optimal_ratio"]) for row in exact_rows
    )
    ratio_upper = max(1.2, ceil((observed_maximum + 0.04) * 10) / 10)
    bound_values = [
        (K, log(K * (K - 1) / 2) + 1)
        for K in k_values
    ]
    bound_upper = float(ceil(max(value for _, value in bound_values)))

    width, height = 1120, 730
    panels = (
        (90.0, 105.0, 390.0, 205.0),
        (635.0, 105.0, 390.0, 205.0),
        (90.0, 425.0, 390.0, 205.0),
        (635.0, 425.0, 390.0, 205.0),
    )
    lines = _svg_document(
        width, height,
        "Greedy Approximation Quality Across Field Families",
    )
    lines.append(
        '<text class="note" x="560" y="54" text-anchor="middle">'
        "Markers: Ngreedy / Noptimal (left axis); purple dotted curve: B(K) = 1 + ln(K(K−1)/2) (right axis).</text>"
    )

    for family, (left, top, panel_width, panel_height) in zip(families, panels):
        family_rows = [row for row in exact_rows if str(row["family"]) == family]
        label = _FAMILY_STYLES[family][1]
        family_ratios = [
            float(row["greedy_to_optimal_ratio"]) for row in family_rows
        ]
        lines.append(
            f'<text class="panel" x="{left + panel_width / 2}" y="{top - 16}" '
            f'text-anchor="middle">{escape(label)}  (median = {median(family_ratios):.2f})</text>'
        )
        _append_axes(
            lines, left=left, top=top, width=panel_width, height=panel_height,
            x_lower=k_lower, x_upper=k_upper, y_lower=0.95, y_upper=ratio_upper,
            x_ticks=[(float(K), str(K)) for K in k_values],
            y_ticks=_ratio_ticks(ratio_upper),
            x_label="Number of fields, K", y_label="Ngreedy / Noptimal",
        )
        _append_bound_axis(
            lines, left=left, top=top, width=panel_width, height=panel_height,
            lower=1, upper=bound_upper,
        )
        bound_points = " ".join(
            f"{_svg_position(K, k_lower, k_upper, left, panel_width):.2f},"
            f"{_svg_position(value, 1, bound_upper, top + panel_height, -panel_height):.2f}"
            for K, value in bound_values
        )
        lines.append(f'<polyline class="bound" points="{bound_points}"/>')
        equality_y = _svg_position(
            1.0, 0.95, ratio_upper, top + panel_height, -panel_height
        )
        lines.append(
            f'<line class="diagonal" x1="{left}" y1="{equality_y:.2f}" '
            f'x2="{left + panel_width}" y2="{equality_y:.2f}"/>'
        )
        family_m_values = sorted({int(row["M"]) for row in family_rows})
        minimum_m, maximum_m = family_m_values[0], family_m_values[-1]
        for row in family_rows:
            if maximum_m == minimum_m:
                offset = 0.0
            else:
                offset = 0.26 * (
                    (int(row["M"]) - minimum_m) / (maximum_m - minimum_m) - 0.5
                )
            x = _svg_position(
                float(row["F"]) + offset, k_lower, k_upper, left, panel_width
            )
            y = _svg_position(
                float(row["greedy_to_optimal_ratio"]),
                0.95, ratio_upper, top + panel_height, -panel_height,
            )
            _append_marker(lines, x, y, family, size=4.5)

    lines.append(
        '<text class="note" x="560" y="700" text-anchor="middle">'
        "Grey dashed: Ngreedy = Noptimal. Each marker is one instance; small horizontal offsets separate different M values.</text>"
    )
    lines.append("</svg>")
    (summary_output / "cross_family_quality.svg").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_cross_family_summary(
    rows: list[dict[str, int | float | str | None]],
) -> None:
    """Write aggregate numerical and visual summaries after an all-dataset run."""
    exact_rows = [row for row in rows if row["Nmin_optimal"] is not None]
    if len({str(row["family"]) for row in exact_rows}) < 2:
        return
    summary_output = OUTPUT_ROOT / "cross-family-summary"
    summary_output.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_output / "case_results.csv", rows)
    _write_family_summary(summary_output / "family_summary.csv", rows)
    _write_cross_family_quality_figure(summary_output, rows)


def _write_greedy_scalability_figure(
    dataset_output: Path,
    rows: list[dict[str, int | float | str | None]],
) -> None:
    """Write controlled large-K figures without claiming an unknown optimum."""
    greedy_rows = [row for row in rows if row["Nmin_greedy"] is not None]
    if not greedy_rows:
        return

    width, height = 980, 440
    left_panel = (80.0, 80.0, 340.0, 270.0)
    right_panel = (585.0, 80.0, 340.0, 270.0)

    # Select the richest fixed-M and fixed-K slices so the two panels do not
    # conflate field-count and location-count effects.
    k_sweep_rows, fixed_M = _controlled_k_sweep(greedy_rows)
    m_sweep_rows, fixed_K = _controlled_m_sweep(greedy_rows)
    k_values = sorted({int(row["F"]) for row in k_sweep_rows})
    m_values = sorted({int(row["M"]) for row in m_sweep_rows})
    n_upper = float(ceil(max(float(row["Nmin_greedy"]) for row in m_sweep_rows) / 5) * 5)
    n_ticks = _integer_ticks(n_upper)
    runtime_ms = [
        float(row["greedy_seconds"]) * 1_000
        for row in k_sweep_rows
        if row["greedy_seconds"] is not None
    ]
    time_upper = ceil(max(runtime_ms) * 1.2 * 10) / 10
    time_ticks = [(time_upper * tick / 4, f"{time_upper * tick / 4:.1f}") for tick in range(5)]
    lines = _svg_document(width, height, "Perfect Pairwise Discrimination: Greedy Scalability")
    _append_family_legend(lines, 360, 51, _present_families(greedy_rows))

    left, top, panel_width, panel_height = left_panel
    lines.append(f'<text class="panel" x="{left + panel_width / 2}" y="63" text-anchor="middle">Runtime: fixed M = {fixed_M}</text>')
    _append_axes(
        lines, left=left, top=top, width=panel_width, height=panel_height,
        x_lower=min(k_values) - 0.5, x_upper=max(k_values) + 0.5,
        y_lower=0, y_upper=time_upper,
        x_ticks=[(float(K), str(K)) for K in k_values], y_ticks=time_ticks,
        x_label="Number of fields, K", y_label="Mean runtime per run (ms)",
    )
    family_offsets = {"gaussian": -0.18, "sine": -0.06, "fourier": 0.06, "polynomial": 0.18}
    for row in k_sweep_rows:
        x_value = float(row["F"]) + family_offsets.get(str(row["family"]), 0.0)
        x = _svg_position(x_value, min(k_values) - 0.5, max(k_values) + 0.5, left, panel_width)
        y = _svg_position(float(row["greedy_seconds"]) * 1_000, 0, time_upper, top + panel_height, -panel_height)
        _append_marker(lines, x, y, str(row["family"]))

    left, top, panel_width, panel_height = right_panel
    lines.append(f'<text class="panel" x="{left + panel_width / 2}" y="63" text-anchor="middle">Sensor requirement: fixed K = {fixed_K}</text>')
    _append_axes(
        lines, left=left, top=top, width=panel_width, height=panel_height,
        x_lower=min(m_values) - 0.5, x_upper=max(m_values) + 0.5,
        y_lower=0, y_upper=n_upper,
        x_ticks=[(float(M), str(M)) for M in m_values], y_ticks=n_ticks,
        x_label="Number of locations, M", y_label="Nmin (greedy)",
    )
    for row in m_sweep_rows:
        x_value = float(row["M"]) + family_offsets.get(str(row["family"]), 0.0)
        x = _svg_position(x_value, min(m_values) - 0.5, max(m_values) + 0.5, left, panel_width)
        y = _svg_position(float(row["Nmin_greedy"]), 0, n_upper, top + panel_height, -panel_height)
        _append_marker(lines, x, y, str(row["family"]))

    lines.append('<text class="note" x="490" y="418" text-anchor="middle">Fixed-parameter slices isolate K and M; colours identify the field family.</text>')
    lines.append("</svg>")
    (dataset_output / "greedy_scalability.svg").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_paper_figure(
    dataset_output: Path,
    rows: list[dict[str, int | float | str | None]],
) -> None:
    """Write the appropriate compact figure and remove the legacy long plot."""
    legacy_plot = dataset_output / "nmin_comparison.svg"
    legacy_plot.unlink(missing_ok=True)
    if any(row["Nmin_optimal"] is not None for row in rows):
        _write_algorithm_quality_figure(dataset_output, rows)
    else:
        _write_greedy_scalability_figure(dataset_output, rows)


def _evaluate_dataset(
    dataset_directory: Path, requested_mode: str, repetitions: int
) -> tuple[bool, list[dict[str, int | float | str | None]]]:
    input_files = sorted(dataset_directory.glob("*.csv"))
    if not input_files:
        print(f"No CSV files found in: {dataset_directory}", file=sys.stderr)
        return False, []
    case_families = _case_families_from_manifest(dataset_directory)

    # Reject an incompatible request before creating or changing any outputs.
    if requested_mode in ("both", "optimal"):
        for input_file in input_files:
            F, _, _ = load_rijm_csv(input_file)
            if F > 10:
                raise ValueError(
                    f"Dataset '{dataset_directory.name}' contains {input_file.name} "
                    f"with K={F}. Exact mode supports only K <= 10; use --mode greedy or --mode auto."
                )

    dataset_output = OUTPUT_ROOT / dataset_directory.name
    dataset_output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | float | str | None]] = []
    errors: list[dict[str, str]] = []

    for input_file in input_files:
        try:
            F, M, rijm = load_rijm_csv(input_file)
            Nijm = compute_nijm(F, M, rijm)
            mode = _mode_for_case(requested_mode, F)
            record = _build_record(
                input_file.stem, F, M, rijm, Nijm, mode, repetitions
            )
            family = case_families.get(
                input_file.stem, _family_from_case_name(input_file.stem)
            )
            record["field_family"] = family
            rows.append(_write_case_output(dataset_output, record, family))
            print(f"Processed {dataset_directory.name}/{input_file.name} ({mode})")
        except (OSError, TypeError, ValueError) as error:
            errors.append({"file": input_file.name, "error": str(error)})
            print(f"Skipped {input_file.name}: {error}", file=sys.stderr)

    if rows:
        _write_csv(dataset_output / "nmin_comparison.csv", rows)
        _write_family_summary(dataset_output / "family_summary.csv", rows)
        _write_paper_figure(dataset_output, rows)
        _print_table(dataset_directory.name, rows)

    if errors:
        with (dataset_output / "errors.json").open("w", encoding="utf-8") as output_file:
            json.dump(errors, output_file, indent=2)
            output_file.write("\n")

    return bool(rows) and not errors, rows


def main() -> int:
    arguments = _parse_arguments()
    try:
        datasets = _dataset_directories(arguments.dataset)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        evaluations = [
            _evaluate_dataset(dataset, arguments.mode, arguments.repeats)
            for dataset in datasets
        ]
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if arguments.dataset == "all":
        _write_cross_family_summary(
            [row for _, rows in evaluations for row in rows]
        )
    succeeded = [successful for successful, _ in evaluations]
    return 0 if all(succeeded) else 1


if __name__ == "__main__":
    raise SystemExit(main())
