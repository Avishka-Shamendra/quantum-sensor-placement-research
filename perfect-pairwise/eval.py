"""Evaluate perfect-pairwise algorithms on all CSV cases in ``inputs/``.

Run from the repository root with:

    python3 perfect-pairwise/eval.py

Each input CSV must use the columns ``i,j,m,r_ijm``.  Results are written to
``perfect-pairwise/output/<case-name>/`` and the overall comparison is written
to ``perfect-pairwise/output/``.
"""

import csv
import json
import sys
from html import escape
from math import log
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
ROOT_DIRECTORY = SCRIPT_DIRECTORY.parent
INPUT_DIRECTORY = ROOT_DIRECTORY / "inputs"
OUTPUT_DIRECTORY = SCRIPT_DIRECTORY / "output"

# ``util.py`` is intentionally at the repository root, while this evaluator
# and the two algorithms live in the ``perfect-pairwise`` directory.
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

from util import compute_nijm, load_rijm_csv  # noqa: E402
from greedy import greedy_perfect_pairwise_sensor_placement  # noqa: E402
from optimal import optimal_perfect_pairwise_sensor_placement  # noqa: E402


def _pair_keys_for_json(values: dict[tuple[int, int], list[int | float]]) -> dict[str, list[int | float]]:
    """Make tuple-keyed data JSON serializable."""
    return {f"{i},{j}": value for (i, j), value in values.items()}


def _result_record(
    case_name: str,
    F: int,
    M: int,
    rijm: dict[tuple[int, int], list[float]],
    Nijm: dict[tuple[int, int], list[int]],
    optimal_result: tuple[int, list[int], list[list[int | None]]],
    greedy_result: tuple[int, list[int], list[list[int | None]]],
) -> dict[str, object]:
    """Build one JSON-ready evaluation record."""
    optimal_total, optimal_allocation, optimal_locations = optimal_result
    greedy_total, greedy_allocation, greedy_locations = greedy_result
    pair_count = F * (F - 1) // 2
    approximation_bound = (log(pair_count) + 1) * optimal_total
    return {
        "case": case_name,
        "F": F,
        "M": M,
        "pair_count": pair_count,
        "rijm": _pair_keys_for_json(rijm),
        "Nijm": _pair_keys_for_json(Nijm),
        "optimal": {
            "Nmin": optimal_total,
            "allocation": optimal_allocation,
            "pair_locations": optimal_locations,
        },
        "greedy": {
            "Nmin": greedy_total,
            "allocation": greedy_allocation,
            "pair_locations": greedy_locations,
        },
        "greedy_approximation_bound": approximation_bound,
    }


def _write_case_output(record: dict[str, object]) -> dict[str, int | float | str]:
    """Write JSON and a compact comparison table for one input case."""
    case_name = str(record["case"])
    case_directory = OUTPUT_DIRECTORY / case_name
    case_directory.mkdir(parents=True, exist_ok=True)

    with (case_directory / "results.json").open("w", encoding="utf-8") as output_file:
        json.dump(record, output_file, indent=2)
        output_file.write("\n")

    optimal = record["optimal"]
    greedy = record["greedy"]
    assert isinstance(optimal, dict) and isinstance(greedy, dict)
    optimal_total = int(optimal["Nmin"])
    greedy_total = int(greedy["Nmin"])
    approximation_bound = float(record["greedy_approximation_bound"])

    with (case_directory / "comparison.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "case",
                "F",
                "M",
                "Nmin_optimal",
                "Nmin_greedy",
                "greedy_approximation_bound",
                "greedy_minus_optimal",
            ]
        )
        writer.writerow(
            [
                case_name,
                record["F"],
                record["M"],
                optimal_total,
                greedy_total,
                approximation_bound,
                greedy_total - optimal_total,
            ]
        )

    return {
        "case": case_name,
        "F": int(record["F"]),
        "M": int(record["M"]),
        "Nmin_optimal": optimal_total,
        "Nmin_greedy": greedy_total,
        "greedy_approximation_bound": approximation_bound,
    }


def _write_overall_table(rows: list[dict[str, int | float | str]]) -> None:
    """Write the cross-case Nmin comparison table."""
    with (OUTPUT_DIRECTORY / "nmin_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "case",
                "F",
                "M",
                "Nmin_optimal",
                "Nmin_greedy",
                "greedy_approximation_bound",
                "greedy_minus_optimal",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["case"],
                    row["F"],
                    row["M"],
                    row["Nmin_optimal"],
                    row["Nmin_greedy"],
                    row["greedy_approximation_bound"],
                    int(row["Nmin_greedy"]) - int(row["Nmin_optimal"]),
                ]
            )


def _print_overall_table(rows: list[dict[str, int | float | str]]) -> None:
    """Print the Nmin comparison table to the terminal."""
    headers = ["Case", "F", "M", "Nmin optimal", "Nmin greedy", "Approx. bound"]
    table_rows = [
        [
            str(row["case"]),
            str(row["F"]),
            str(row["M"]),
            str(row["Nmin_optimal"]),
            str(row["Nmin_greedy"]),
            f"{float(row['greedy_approximation_bound']):.3f}",
        ]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[column]) for row in table_rows))
        for column, header in enumerate(headers)
    ]

    def format_row(row: list[str]) -> str:
        return " | ".join(
            value.rjust(width) for value, width in zip(row, widths)
        )

    print()
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(format_row(row))


def _write_overall_graph(rows: list[dict[str, int | float | str]]) -> None:
    """Write a dependency-free SVG line graph comparing Nmin by case."""
    chart_width = max(760, 150 * len(rows) + 140)
    chart_height = 480
    left, right, top, bottom = 80, 35, 50, 125
    plot_width = chart_width - left - right
    plot_height = chart_height - top - bottom
    maximum = max(float(row["Nmin_optimal"]) for row in rows)
    maximum = max(
        maximum,
        max(float(row["Nmin_greedy"]) for row in rows),
        max(float(row["greedy_approximation_bound"]) for row in rows),
        1,
    )
    y_limit = maximum + max(1, maximum / 10)

    def x_position(index: int) -> float:
        return left + plot_width * (index + 0.5) / len(rows)

    def y_position(value: float) -> float:
        return top + plot_height * (1 - value / y_limit)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{chart_width}" height="{chart_height}" viewBox="0 0 {chart_width} {chart_height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.axis{stroke:#5f6368;stroke-width:1}.grid{stroke:#dadce0;stroke-width:1}.optimal{stroke:#1a73e8;fill:none;stroke-width:3}.greedy{stroke:#d93025;fill:none;stroke-width:3}.bound{stroke:#188038;fill:none;stroke-width:3;stroke-dasharray:8 6}.label{font-size:13px}.title{font-size:18px;font-weight:bold}</style>',
        f'<text class="title" x="{chart_width / 2}" y="26" text-anchor="middle">Nmin and Greedy Approximation Bound by Input Case</text>',
    ]

    for tick in range(6):
        value = y_limit * tick / 5
        y = y_position(value)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{chart_width - right}" y2="{y:.2f}"/>')
        lines.append(f'<text class="label" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{value:.1f}</text>')

    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{chart_height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{chart_height - bottom}" x2="{chart_width - right}" y2="{chart_height - bottom}"/>',
            f'<text class="label" x="22" y="{top + plot_height / 2}" transform="rotate(-90 22 {top + plot_height / 2})" text-anchor="middle">Nmin</text>',
        ]
    )

    for index, row in enumerate(rows):
        x = x_position(index)
        name = escape(str(row["case"]))
        lines.append(
            f'<text class="label" x="{x:.2f}" y="{chart_height - bottom + 20}" text-anchor="end" transform="rotate(-35 {x:.2f} {chart_height - bottom + 20})">{name}</text>'
        )

    for key, css_class, color in (
        ("Nmin_optimal", "optimal", "#1a73e8"),
        ("Nmin_greedy", "greedy", "#d93025"),
        ("greedy_approximation_bound", "bound", "#188038"),
    ):
        points = " ".join(
            f"{x_position(index):.2f},{y_position(float(row[key])):.2f}"
            for index, row in enumerate(rows)
        )
        lines.append(f'<polyline class="{css_class}" points="{points}"/>')
        for index, row in enumerate(rows):
            x = x_position(index)
            y = y_position(float(row[key]))
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')

    legend_x = chart_width - right - 330
    lines.extend(
        [
            f'<line class="optimal" x1="{legend_x}" y1="40" x2="{legend_x + 28}" y2="40"/>',
            f'<text class="label" x="{legend_x + 36}" y="45">Optimal</text>',
            f'<line class="greedy" x1="{legend_x + 100}" y1="40" x2="{legend_x + 128}" y2="40"/>',
            f'<text class="label" x="{legend_x + 136}" y="45">Greedy</text>',
            f'<line class="bound" x1="{legend_x + 200}" y1="40" x2="{legend_x + 228}" y2="40"/>',
            f'<text class="label" x="{legend_x + 236}" y="45">Bound</text>',
            "</svg>",
        ]
    )
    (OUTPUT_DIRECTORY / "nmin_comparison.svg").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    """Evaluate all input cases and create the output artifacts."""
    if not INPUT_DIRECTORY.is_dir():
        print(f"Inputs directory does not exist: {INPUT_DIRECTORY}", file=sys.stderr)
        return 1

    input_files = sorted(INPUT_DIRECTORY.glob("*.csv"))
    if not input_files:
        print(f"No CSV files found in: {INPUT_DIRECTORY}", file=sys.stderr)
        return 1

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | float | str]] = []
    errors: list[dict[str, str]] = []

    for input_file in input_files:
        try:
            F, M, rijm = load_rijm_csv(input_file)
            Nijm = compute_nijm(F, M, rijm)
            optimal_result = optimal_perfect_pairwise_sensor_placement(F, M, Nijm)
            greedy_result = greedy_perfect_pairwise_sensor_placement(F, M, Nijm)
            record = _result_record(
                input_file.stem,
                F,
                M,
                rijm,
                Nijm,
                optimal_result,
                greedy_result,
            )
            rows.append(_write_case_output(record))
            print(f"Processed {input_file.name}")
        except (OSError, TypeError, ValueError) as error:
            errors.append({"file": input_file.name, "error": str(error)})
            print(f"Skipped {input_file.name}: {error}", file=sys.stderr)

    if rows:
        _write_overall_table(rows)
        _write_overall_graph(rows)
        _print_overall_table(rows)

    if errors:
        with (OUTPUT_DIRECTORY / "errors.json").open("w", encoding="utf-8") as output_file:
            json.dump(errors, output_file, indent=2)
            output_file.write("\n")

    return 0 if rows and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
