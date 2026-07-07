"""Evaluate optimal and greedy minimum-error placement algorithms."""

import csv
import json
from html import escape
from pathlib import Path

from greedy_heurisitic import greedy_min_error_pairwise_sensor_placement
from optimal_algo import optimal_min_error_pairwise_sensor_placement


BASE_DIRECTORY = Path(__file__).resolve().parent
INPUT_DIRECTORY = BASE_DIRECTORY / "inputs"
OUTPUT_DIRECTORY = BASE_DIRECTORY / "outputs"


def load_input(file_path):
    """Load one evaluation case from a JSON file."""
    with file_path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    try:
        K = data["K"]
        M = data["M"]
        N = data["N"]
        json_pijmn = data["Pijmn"]
    except KeyError as error:
        raise ValueError(f"Missing required field: {error.args[0]}") from error

    if not isinstance(json_pijmn, dict):
        raise ValueError("Pijmn must be a JSON object.")

    Pijmn = {}

    for pair_key, locations in json_pijmn.items():
        i, j = parse_pair_key(pair_key)

        if not isinstance(locations, dict):
            raise ValueError(f'Pijmn["{pair_key}"] must be a JSON object.')

        for location_key, counts in locations.items():
            try:
                m = int(location_key)
            except ValueError as error:
                raise ValueError(
                    f'Invalid location key "{location_key}".'
                ) from error

            if not isinstance(counts, dict):
                raise ValueError(
                    f'Pijmn["{pair_key}"]["{location_key}"] must be an object.'
                )

            for count_key, error_probability in counts.items():
                try:
                    n = int(count_key)
                except ValueError as error:
                    raise ValueError(
                        f'Invalid sensor-count key "{count_key}".'
                    ) from error

                Pijmn[(i, j, m, n)] = error_probability

    return {
        "name": data.get("name", file_path.stem),
        "K": K,
        "M": M,
        "N": N,
        "Pijmn": Pijmn,
    }


def parse_pair_key(pair_key):
    """Convert a JSON key like '0,2' into the tuple (0, 2)."""
    parts = pair_key.split(",")

    if len(parts) != 2:
        raise ValueError(f'Invalid pair key "{pair_key}". Use "i,j".')

    try:
        i = int(parts[0].strip())
        j = int(parts[1].strip())
    except ValueError as error:
        raise ValueError(
            f'Invalid pair key "{pair_key}". Pair indices must be integers.'
        ) from error

    if i == j:
        raise ValueError(f'Invalid pair key "{pair_key}". Indices must differ.')

    return (i, j) if i < j else (j, i)


def evaluate_case(case):
    """Run both algorithms for one case."""
    optimal_result = optimal_min_error_pairwise_sensor_placement(
        case["K"], case["M"], case["N"], case["Pijmn"]
    )
    greedy_result = greedy_min_error_pairwise_sensor_placement(
        case["K"], case["M"], case["N"], case["Pijmn"]
    )

    return optimal_result, greedy_result


def draw_tree_svg(tree, file_path, title):
    """Draw a tree dictionary as a small dependency-free SVG file."""
    leaf_x = 0
    positions = {}
    nodes = []
    edges = []

    def visit(node, depth):
        nonlocal leaf_x
        node_id = len(nodes)
        nodes.append(node)

        if "declare" in node:
            x = leaf_x * 170 + 85
            leaf_x += 1
            positions[node_id] = (x, depth * 115 + 90)
            return node_id, x

        left_id, left_x = visit(node["discard_i"], depth + 1)
        right_id, right_x = visit(node["discard_j"], depth + 1)
        x = (left_x + right_x) / 2
        positions[node_id] = (x, depth * 115 + 90)
        i, j = node["test_pair"]
        edges.append((node_id, left_id, f"discard f{i}"))
        edges.append((node_id, right_id, f"discard f{j}"))
        return node_id, x

    root_id, _ = visit(tree, 0)
    max_depth = max(len(node["remaining"]) for node in nodes)
    width = max(360, leaf_x * 170)
    height = max(220, max_depth * 115 + 70)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>",
        "text { font-family: Arial, sans-serif; font-size: 13px; }",
        ".title { font-size: 18px; font-weight: 700; }",
        ".node { fill: #f8fafc; stroke: #334155; stroke-width: 1.4; }",
        ".leaf { fill: #ecfdf5; stroke: #047857; stroke-width: 1.4; }",
        ".edge { stroke: #64748b; stroke-width: 1.2; fill: none; }",
        ".edge-label { fill: #475569; font-size: 11px; }",
        "</style>",
        f'<text class="title" x="{width / 2}" y="28" text-anchor="middle">'
        f"{escape(title)}</text>",
    ]

    for parent_id, child_id, label in edges:
        x1, y1 = positions[parent_id]
        x2, y2 = positions[child_id]
        label_x = (x1 + x2) / 2
        label_y = (y1 + y2) / 2 - 6
        parts.append(
            f'<line class="edge" x1="{x1}" y1="{y1 + 22}" '
            f'x2="{x2}" y2="{y2 - 22}" />'
        )
        parts.append(
            f'<text class="edge-label" x="{label_x}" y="{label_y}" '
            f'text-anchor="middle">{escape(label)}</text>'
        )

    for node_id, node in enumerate(nodes):
        x, y = positions[node_id]

        if "declare" in node:
            label = f'declare f{node["declare"]}'
            class_name = "leaf"
        else:
            i, j = node["test_pair"]
            label = f"test f{i} vs f{j}"
            class_name = "node"

        parts.append(
            f'<rect class="{class_name}" x="{x - 55}" y="{y - 22}" '
            'width="110" height="44" rx="6" />'
        )
        parts.append(
            f'<text x="{x}" y="{y + 5}" text-anchor="middle">'
            f"{escape(label)}</text>"
        )

    # Keep the root id referenced so accidental tree format changes are obvious.
    if root_id != 0:
        raise ValueError("Unexpected tree traversal order.")

    parts.append("</svg>")
    file_path.write_text("\n".join(parts), encoding="utf-8")


def input_sort_key(file_path):
    """Sort input_1.json through input_10.json numerically."""
    suffix = file_path.stem.removeprefix("input_")
    if suffix.isdigit():
        return 0, int(suffix)
    return 1, suffix


def format_allocation(allocation):
    return "[" + ", ".join(str(count) for count in allocation) + "]"


def print_table(rows):
    """Print all evaluation results as an aligned table."""
    headers = [
        "Input",
        "K",
        "M",
        "N",
        "Optimal Error",
        "Greedy Error",
        "Optimal Allocation",
        "Greedy Allocation",
    ]
    string_rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in string_rows))
        for column in range(len(headers))
    ]

    def format_row(row):
        return " | ".join(
            value.rjust(width)
            for value, width in zip(row, widths)
        )

    lines = [format_row(headers)]
    lines.append("-+-".join("-" * width for width in widths))
    lines.extend(format_row(row) for row in string_rows)
    table = "\n".join(lines)
    print(table)
    return table


def save_summary_files(rows, table):
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    (OUTPUT_DIRECTORY / "comparison_table.txt").write_text(
        table + "\n", encoding="utf-8"
    )

    with (OUTPUT_DIRECTORY / "comparison_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "input",
                "K",
                "M",
                "N",
                "optimal_error",
                "greedy_error",
                "optimal_allocation",
                "greedy_allocation",
            ]
        )
        writer.writerows(rows)


def main():
    """Evaluate every JSON file in the inputs directory."""
    input_files = sorted(INPUT_DIRECTORY.glob("*.json"), key=input_sort_key)

    if not input_files:
        print(f"No JSON input files found in: {INPUT_DIRECTORY}")
        return 1

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    rows = []
    errors = []

    for file_path in input_files:
        try:
            case = load_input(file_path)
            optimal_result, greedy_result = evaluate_case(case)
            optimal_error, optimal_allocation, _, optimal_tree = optimal_result
            greedy_error, greedy_allocation, _, greedy_tree = greedy_result
            input_number = file_path.stem.removeprefix("input_")

            draw_tree_svg(
                optimal_tree,
                OUTPUT_DIRECTORY / f"{file_path.stem}_optimal_tree.svg",
                f"{case['name']} optimal tree",
            )
            draw_tree_svg(
                greedy_tree,
                OUTPUT_DIRECTORY / f"{file_path.stem}_greedy_tree.svg",
                f"{case['name']} greedy tree",
            )

            rows.append(
                [
                    input_number,
                    case["K"],
                    case["M"],
                    case["N"],
                    f"{optimal_error:.6f}",
                    f"{greedy_error:.6f}",
                    format_allocation(optimal_allocation),
                    format_allocation(greedy_allocation),
                ]
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            errors.append((file_path.name, error))

    if rows:
        table = print_table(rows)
        save_summary_files(rows, table)

    for file_name, error in errors:
        print(f"\nERROR in {file_name}: {error}")

    if errors:
        return 1

    print(f"\nSaved tree SVGs and comparison tables in: {OUTPUT_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
