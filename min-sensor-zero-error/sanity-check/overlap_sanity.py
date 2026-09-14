"""Appendix-state overlap sanity checks for zero-error benchmark inputs.

This module is intentionally self-contained so the sanity-check notebook can be
run from this folder without depending on the main solver implementation.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from cmath import exp
from dataclasses import dataclass
from math import ceil, cos, isclose, isfinite, pi, sqrt
from numbers import Real
from pathlib import Path
from statistics import fmean


PhaseGaps = dict[tuple[int, int], list[float]]


@dataclass(frozen=True)
class AppendixOverlap:
    """One overlap evaluation for one pair and sensing location."""

    threshold: int | None
    construction_case: str
    phase_spread: float | None
    overlap_real: float | None
    overlap_imag: float | None
    overlap_abs: float | None
    support_size: int
    state_description: str


def principal_phase(value: Real) -> float:
    """Wrap a finite phase to the principal interval (-pi, pi]."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("phase must be a finite real number")
    wrapped = (float(value) + pi) % (2 * pi) - pi
    return pi if wrapped <= -pi else wrapped


def minimum_sensors_for_phase_gap(r_ijm: Real) -> int | None:
    """Return ceil(pi / abs(r_ijm)), or None for an unusable zero gap."""
    phase = principal_phase(r_ijm)
    if phase == 0.0:
        return None
    return ceil(pi / abs(phase))


def load_phase_csv(csv_path: str | Path) -> tuple[int, int, PhaseGaps]:
    """Load one benchmark case with rows i,j,m,r_ijm."""
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"i", "j", "m", "r_ijm"}
        if reader.fieldnames is None or set(reader.fieldnames) != required_columns:
            raise ValueError(f"{path} must have exactly these columns: i,j,m,r_ijm")

        raw: dict[tuple[int, int], dict[int, float]] = {}
        for row_number, row in enumerate(reader, start=2):
            try:
                i = int(row["i"])
                j = int(row["j"])
                m = int(row["m"])
                phase = float(row["r_ijm"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid value on row {row_number} of {path}") from error

            if i < 0 or j < 0 or i >= j or m < 0:
                raise ValueError(
                    f"Row {row_number} of {path} must satisfy 0 <= i < j and m >= 0"
                )
            if not isfinite(phase) or not -pi < phase <= pi:
                raise ValueError(
                    f"r_ijm on row {row_number} must lie in (-pi, pi]"
                )
            raw.setdefault((i, j), {})[m] = phase

    if not raw:
        raise ValueError(f"CSV file contains no phase data: {path}")

    K = max(j for _, j in raw) + 1
    M = max(m for values in raw.values() for m in values) + 1
    pairs = tuple((i, j) for i in range(K) for j in range(i + 1, K))
    if set(raw) != set(pairs):
        raise ValueError(f"{path} must contain every canonical field pair")

    phase_gaps: PhaseGaps = {}
    for pair in pairs:
        if set(raw[pair]) != set(range(M)):
            raise ValueError(
                f"{path} must contain locations 0 through {M - 1} for pair {pair}"
            )
        phase_gaps[pair] = [principal_phase(raw[pair][m]) for m in range(M)]
    return K, M, phase_gaps


def appendix_state_overlap(
    r_ijm: Real,
    *,
    boundary_tolerance: float = 1e-12,
) -> AppendixOverlap:
    """Evaluate <psi|V_ijm^{tensor n}|psi> for the Appendix construction.

    The state is represented by its nonzero eigen-components instead of a full
    2^n vector. This is essential because some generated phase gaps are tiny,
    and therefore their thresholds n = N_ijm can be extremely large.
    """
    phase = principal_phase(r_ijm)
    n = minimum_sensors_for_phase_gap(phase)
    if n is None:
        return AppendixOverlap(
            threshold=None,
            construction_case="zero_phase_gap_unusable",
            phase_spread=None,
            overlap_real=None,
            overlap_imag=None,
            overlap_abs=None,
            support_size=0,
            state_description="No finite Appendix state exists because r_ijm = 0.",
        )

    theta = abs(phase) / 2.0
    n_theta = n * theta
    phase_spread = n * abs(phase)

    if isclose(phase_spread, pi, rel_tol=0.0, abs_tol=boundary_tolerance):
        components = (
            (0.5, n_theta),
            (0.5, -n_theta),
        )
        construction_case = "boundary"
        state_description = "(|A+> + |A->) / sqrt(2)"
    elif n % 2 == 0:
        w = 1.0 / (1.0 - cos(n_theta))
        components = (
            (w / 2.0, n_theta),
            (w / 2.0, -n_theta),
            (1.0 - w, 0.0),
        )
        construction_case = "above_threshold_even_n"
        state_description = (
            "sqrt(w/2)|A+> + sqrt(w/2)|A-> + sqrt(1-w)|B0>"
        )
    else:
        w = cos(theta) / (cos(theta) - cos(n_theta))
        components = (
            (w / 2.0, n_theta),
            (w / 2.0, -n_theta),
            ((1.0 - w) / 2.0, theta),
            ((1.0 - w) / 2.0, -theta),
        )
        construction_case = "above_threshold_odd_n"
        state_description = (
            "sqrt(w/2)|A+> + sqrt(w/2)|A-> "
            "+ sqrt((1-w)/2)|B+> + sqrt((1-w)/2)|B->"
        )

    overlap = sum(probability * exp(1j * eigenphase) for probability, eigenphase in components)
    return AppendixOverlap(
        threshold=n,
        construction_case=construction_case,
        phase_spread=phase_spread,
        overlap_real=overlap.real,
        overlap_imag=overlap.imag,
        overlap_abs=abs(overlap),
        support_size=len(components),
        state_description=state_description,
    )


def case_paths(input_root: str | Path, dataset: str) -> list[Path]:
    """Return case CSV paths in manifest order when a manifest exists."""
    dataset_directory = Path(input_root) / dataset
    manifest_path = dataset_directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [dataset_directory / str(case["file"]) for case in manifest["cases"]]
    return sorted(dataset_directory.glob("case_*.csv"))


def iter_overlap_rows(
    input_root: str | Path,
    dataset: str,
    *,
    case_limit: int | None = None,
    boundary_tolerance: float = 1e-12,
    progress_every: int | None = 250,
) -> Iterable[dict[str, object]]:
    """Yield one overlap row for every finite or zero-gap pair/location."""
    paths = case_paths(input_root, dataset)
    if case_limit is not None:
        paths = paths[:case_limit]
    total_cases = len(paths)

    for case_number, path in enumerate(paths, start=1):
        K, M, phase_gaps = load_phase_csv(path)
        for (i, j), phases in phase_gaps.items():
            for m, phase in enumerate(phases):
                result = appendix_state_overlap(
                    phase,
                    boundary_tolerance=boundary_tolerance,
                )
                yield {
                    "dataset": dataset,
                    "case": path.stem,
                    "case_number": case_number,
                    "K": K,
                    "M": M,
                    "i": i,
                    "j": j,
                    "m": m,
                    "r_ijm": phase,
                    "N_ijm": "inf" if result.threshold is None else result.threshold,
                    "construction_case": result.construction_case,
                    "phase_spread": result.phase_spread,
                    "overlap_real": result.overlap_real,
                    "overlap_imag": result.overlap_imag,
                    "overlap_abs": result.overlap_abs,
                    "support_size": result.support_size,
                    "state_description": result.state_description,
                }

        if (
            progress_every is not None
            and progress_every > 0
            and (
                case_number == 1
                or case_number % progress_every == 0
                or case_number == total_cases
            )
        ):
            print(f"{dataset}: processed case {case_number}/{total_cases}", flush=True)


def run_sanity_check(
    input_root: str | Path,
    output_root: str | Path,
    *,
    datasets: Sequence[str] = ("benchmark_1d", "benchmark_2d"),
    case_limit: int | None = None,
    overlap_tolerance: float = 1e-12,
    boundary_tolerance: float = 1e-12,
    progress_every: int | None = 250,
    write_detailed_csv: bool = False,
    sample_size: int = 20,
) -> dict[str, object]:
    """Run the Appendix overlap check and write summary/violation tables."""
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    violation_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for dataset in datasets:
        print(f"Checking {dataset} ...", flush=True)
        finite_overlaps: list[float] = []
        zero_gap_locations = 0
        cases_seen: set[str] = set()
        construction_counts: Counter[str] = Counter()
        worst_row: dict[str, object] | None = None
        detail_file = None
        detail_writer = None

        if write_detailed_csv:
            detail_path = output / f"{dataset}_all_pair_location_overlaps.csv"
            detail_file = detail_path.open("w", encoding="utf-8", newline="")

        try:
            for row in iter_overlap_rows(
                input_root,
                dataset,
                case_limit=case_limit,
                boundary_tolerance=boundary_tolerance,
                progress_every=progress_every,
            ):
                cases_seen.add(str(row["case"]))
                construction_counts[str(row["construction_case"])] += 1

                if detail_file is not None:
                    if detail_writer is None:
                        detail_writer = csv.DictWriter(
                            detail_file,
                            fieldnames=list(row),
                            lineterminator="\n",
                        )
                        detail_writer.writeheader()
                    detail_writer.writerow(row)

                if len(sample_rows) < sample_size:
                    sample_rows.append(row)

                overlap_abs = row["overlap_abs"]
                if overlap_abs is None:
                    zero_gap_locations += 1
                    continue

                overlap_abs = float(overlap_abs)
                finite_overlaps.append(overlap_abs)
                if worst_row is None or overlap_abs > float(worst_row["overlap_abs"]):
                    worst_row = row
                if overlap_abs > overlap_tolerance:
                    violation_rows.append(row)
        finally:
            if detail_file is not None:
                detail_file.close()

        if not finite_overlaps:
            raise ValueError(f"No finite phase gaps were found for {dataset}")

        summary_rows.append(
            {
                "dataset": dataset,
                "cases": len(cases_seen),
                "finite_pair_locations": len(finite_overlaps),
                "zero_gap_unusable_locations": zero_gap_locations,
                "max_overlap_abs": max(finite_overlaps),
                "mean_overlap_abs": fmean(finite_overlaps),
                "rms_overlap_abs": sqrt(fmean(value * value for value in finite_overlaps)),
                "violation_count": sum(value > overlap_tolerance for value in finite_overlaps),
                "overlap_tolerance": overlap_tolerance,
                "boundary_cases": construction_counts["boundary"],
                "above_threshold_even_n_cases": construction_counts[
                    "above_threshold_even_n"
                ],
                "above_threshold_odd_n_cases": construction_counts[
                    "above_threshold_odd_n"
                ],
                "worst_case": worst_row["case"] if worst_row else None,
                "worst_pair": (
                    f"({worst_row['i']}, {worst_row['j']})" if worst_row else None
                ),
                "worst_location": worst_row["m"] if worst_row else None,
                "worst_N_ijm": worst_row["N_ijm"] if worst_row else None,
            }
        )

    _write_csv(output / "overlap_summary.csv", summary_rows)
    _write_csv(output / "overlap_violations.csv", violation_rows)
    _write_csv(output / "overlap_sample.csv", sample_rows)

    return {
        "summary": summary_rows,
        "violations": violation_rows,
        "sample": sample_rows,
        "output_root": str(output),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a CSV file, preserving an empty file when there are no rows."""
    with path.open("w", encoding="utf-8", newline="") as output_file:
        if not rows:
            output_file.write("")
            return
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
