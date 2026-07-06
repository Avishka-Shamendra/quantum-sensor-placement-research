"""Compare the optimal and greedy sensor-placement algorithms."""

import json
from math import log
from pathlib import Path

from greedy_algo import greedy_perfect_pairwise_sensor_placement
from optimal_algo import optimal_perfect_pairwise_sensor_placement


INPUT_DIRECTORY = Path(__file__).resolve().parent / "inputs"


def load_input(file_path):
    """Load one evaluation case from a JSON file."""
    with file_path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    try:
        K = data["K"]
        M = data["M"]
        json_nijm = data["Nijm"]
    except KeyError as error:
        raise ValueError(f"Missing required field: {error.args[0]}") from error

    if not isinstance(json_nijm, dict):
        raise ValueError("Nijm must be a JSON object.")

    # JSON keys are strings, so "0,1" is converted to the tuple (0, 1).
    Nijm = {}
    for pair_key, thresholds in json_nijm.items():
        parts = pair_key.split(",")
        if len(parts) != 2:
            raise ValueError(
                f'Invalid pair key "{pair_key}". Use the format "i,j".'
            )

        try:
            pair = (int(parts[0].strip()), int(parts[1].strip()))
        except ValueError as error:
            raise ValueError(
                f'Invalid pair key "{pair_key}". Pair indices must be integers.'
            ) from error

        if pair in Nijm:
            raise ValueError(f"Duplicate function pair: {pair}.")

        Nijm[pair] = thresholds

    return K, M, Nijm


def evaluate_case(K, M, Nijm):
    """Run both algorithms and calculate the theoretical greedy bound."""
    optimal_total, _, _ = (
        optimal_perfect_pairwise_sensor_placement(K, M, Nijm)
    )
    greedy_total, _, _ = (
        greedy_perfect_pairwise_sensor_placement(K, M, Nijm)
    )

    pair_count = K * (K - 1) / 2
    theoretical_bound = (log(pair_count) + 1) * optimal_total

    return optimal_total, greedy_total, theoretical_bound


def input_sort_key(file_path):
    """Sort input_1.json through input_10.json numerically."""
    suffix = file_path.stem.removeprefix("input_")
    if suffix.isdigit():
        return 0, int(suffix)
    return 1, suffix


def print_table(rows):
    """Print all evaluation results as an aligned table."""
    headers = [
        "Input Number",
        "K",
        "M",
        "Noptimal",
        "Ngreedy",
        "(ln(K(K-1)/2)+1)Nopt",
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]

    def format_row(row):
        return " | ".join(
            value.rjust(width)
            for value, width in zip(row, widths)
        )

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def main():
    """Evaluate every JSON file in the inputs directory."""
    input_files = sorted(
        INPUT_DIRECTORY.glob("*.json"),
        key=input_sort_key,
    )

    if not input_files:
        print(f"No JSON input files found in: {INPUT_DIRECTORY}")
        return 1

    rows = []
    errors = []

    for file_path in input_files:
        try:
            K, M, Nijm = load_input(file_path)
            optimal, greedy, bound = evaluate_case(K, M, Nijm)
            input_number = file_path.stem.removeprefix("input_")
            rows.append(
                [
                    input_number,
                    str(K),
                    str(M),
                    str(optimal),
                    str(greedy),
                    f"{bound:.4f}",
                ]
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            errors.append((file_path.name, error))

    if rows:
        print_table(rows)

    for file_name, error in errors:
        print(f"\nERROR in {file_name}: {error}")

    if errors:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
