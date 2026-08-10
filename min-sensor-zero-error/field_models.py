"""Synthetic one- and two-dimensional spatial fields for the benchmarks.

Each generated case contains ``K`` fields evaluated at ``M`` random candidate
locations in either ``[0, 1]`` or ``[0, 1]^2``.  The phase-gap conversion uses
the paper's definition

    r_ijm = Arg(exp(i c (f_i(x_m) - f_j(x_m)))).

Only canonical pairs ``i < j`` are stored.  Reverse phases are reconstructed
with the principal-phase convention, so antisymmetry is maintained modulo
``2 pi``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations
from math import cos, exp, isfinite, pi, sin
from numbers import Real
from random import Random


Location = tuple[float, ...]
FieldValues = list[list[float]]
FieldParameters = list[dict[str, object]]
FieldGeneration = tuple[FieldValues, FieldParameters]
CsvRow = tuple[int, int, int, float]
FieldFactory = Callable[[int, Sequence[Location], Random], FieldGeneration]


def sensing_locations(dimension: int, M: int, rng: Random) -> list[Location]:
    """Sample ``M`` independent uniform locations from ``[0, 1]^dimension``."""
    if dimension not in (1, 2):
        raise ValueError("dimension must be 1 or 2.")
    if isinstance(M, bool) or not isinstance(M, int) or M < 2:
        raise ValueError("M must be an integer of at least 2.")
    return [tuple(rng.random() for _ in range(dimension)) for _ in range(M)]


def _validate_field_request(K: int, locations: Sequence[Location]) -> int:
    if isinstance(K, bool) or not isinstance(K, int) or K < 2:
        raise ValueError("K must be an integer of at least 2.")
    if not locations:
        raise ValueError("locations must not be empty.")
    dimension = len(locations[0])
    if dimension not in (1, 2):
        raise ValueError("locations must be one- or two-dimensional.")
    for location in locations:
        if len(location) != dimension:
            raise ValueError("all locations must have the same dimension.")
        if any(
            isinstance(coordinate, bool)
            or not isinstance(coordinate, Real)
            or not isfinite(coordinate)
            or not 0 <= float(coordinate) <= 1
            for coordinate in location
        ):
            raise ValueError("location coordinates must be finite values in [0, 1].")
    return dimension


def _sample_centre(dimension: int, rng: Random) -> Location:
    return tuple(rng.uniform(0.0, 1.0) for _ in range(dimension))


def _centre_metadata(centre: Location) -> float | list[float]:
    return centre[0] if len(centre) == 1 else list(centre)


def generate_gaussian_fields(
    K: int, locations: Sequence[Location], rng: Random
) -> FieldGeneration:
    """Generate localized isotropic Gaussian fields."""
    dimension = _validate_field_request(K, locations)
    fields: FieldValues = []
    parameters: FieldParameters = []
    for field_index in range(K):
        amplitude = rng.uniform(0.5, 1.5)
        centre = _sample_centre(dimension, rng)
        width = rng.uniform(0.1, 0.3)
        parameters.append({
            "field_index": field_index,
            "amplitude": amplitude,
            "centre": _centre_metadata(centre),
            "width": width,
        })
        fields.append([
            amplitude
            * exp(
                -sum(
                    (coordinate - centre_coordinate) ** 2
                    for coordinate, centre_coordinate in zip(location, centre)
                )
                / (2 * width**2)
            )
            for location in locations
        ])
    return fields, parameters


def generate_sinusoidal_fields(
    K: int, locations: Sequence[Location], rng: Random
) -> FieldGeneration:
    """Generate 1D sinusoids or 2D sinusoidal plane waves."""
    dimension = _validate_field_request(K, locations)
    fields: FieldValues = []
    parameters: FieldParameters = []
    for field_index in range(K):
        amplitude = rng.uniform(0.5, 1.5)
        frequency = rng.randint(1, 3)
        phase_offset = rng.uniform(-pi, pi)
        parameter_record: dict[str, object] = {
            "field_index": field_index,
            "amplitude": amplitude,
            "frequency": frequency,
            "phase_offset": phase_offset,
        }
        if dimension == 1:
            fields.append([
                amplitude
                * sin(2 * pi * frequency * location[0] + phase_offset)
                for location in locations
            ])
        else:
            orientation = rng.uniform(0.0, pi)
            parameter_record["orientation"] = orientation
            fields.append([
                amplitude
                * sin(
                    2
                    * pi
                    * frequency
                    * (
                        location[0] * cos(orientation)
                        + location[1] * sin(orientation)
                    )
                    + phase_offset
                )
                for location in locations
            ])
        parameters.append(parameter_record)
    return fields, parameters


def generate_quadratic_fields(
    K: int, locations: Sequence[Location], rng: Random
) -> FieldGeneration:
    """Generate centered quadratic trends with dimension-normalized scale."""
    dimension = _validate_field_request(K, locations)
    fields: FieldValues = []
    parameters: FieldParameters = []
    for field_index in range(K):
        background = rng.uniform(-0.5, 0.5)
        curvature = rng.uniform(-1.0, 1.0)
        centre = _sample_centre(dimension, rng)
        parameters.append({
            "field_index": field_index,
            "background": background,
            "curvature": curvature,
            "centre": _centre_metadata(centre),
        })
        fields.append([
            background
            + curvature
            / dimension
            * sum(
                (coordinate - centre_coordinate) ** 2
                for coordinate, centre_coordinate in zip(location, centre)
            )
            for location in locations
        ])
    return fields, parameters


FIELD_MODELS: dict[str, FieldFactory] = {
    "gaussian": generate_gaussian_fields,
    "sinusoidal": generate_sinusoidal_fields,
    "quadratic": generate_quadratic_fields,
}

FIELD_FORMULAS: dict[str, dict[int, str]] = {
    "gaussian": {
        1: "f_k(alpha) = A_k exp(-(alpha-mu_k)^2 / (2 w_k^2))",
        2: (
            "f_k(alpha,beta) = A_k exp(-((alpha-mu_k1)^2 + "
            "(beta-mu_k2)^2) / (2 w_k^2))"
        ),
    },
    "sinusoidal": {
        1: "f_k(alpha) = A_k sin(2 pi q_k alpha + gamma_k)",
        2: (
            "f_k(alpha,beta) = A_k sin(2 pi q_k "
            "(alpha cos(theta_k) + beta sin(theta_k)) + gamma_k)"
        ),
    },
    "quadratic": {
        1: "f_k(alpha) = b_k + a_k (alpha-mu_k)^2",
        2: (
            "f_k(alpha,beta) = b_k + (a_k/2) ((alpha-mu_k1)^2 + "
            "(beta-mu_k2)^2)"
        ),
    },
}

FIELD_PARAMETER_RANGES: dict[str, dict[int, dict[str, object]]] = {
    "gaussian": {
        1: {"A_k": [0.5, 1.5], "mu_k": [0.0, 1.0], "w_k": [0.1, 0.3]},
        2: {
            "A_k": [0.5, 1.5],
            "mu_k1": [0.0, 1.0],
            "mu_k2": [0.0, 1.0],
            "w_k": [0.1, 0.3],
        },
    },
    "sinusoidal": {
        1: {
            "A_k": [0.5, 1.5],
            "q_k": [1, 2, 3],
            "gamma_k": [-pi, pi],
        },
        2: {
            "A_k": [0.5, 1.5],
            "q_k": [1, 2, 3],
            "theta_k": [0.0, pi],
            "gamma_k": [-pi, pi],
        },
    },
    "quadratic": {
        1: {"b_k": [-0.5, 0.5], "a_k": [-1.0, 1.0], "mu_k": [0.0, 1.0]},
        2: {
            "b_k": [-0.5, 0.5],
            "a_k": [-1.0, 1.0],
            "mu_k1": [0.0, 1.0],
            "mu_k2": [0.0, 1.0],
        },
    },
}


def principal_phase(value: Real) -> float:
    """Wrap a finite phase to the interval ``(-pi, pi]``."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("phase value must be a finite real number.")
    wrapped = (float(value) + pi) % (2 * pi) - pi
    return pi if wrapped <= -pi else wrapped


def phase_gap_rows(fields: FieldValues, c: Real) -> list[CsvRow]:
    """Convert sampled field values into canonical phase-gap rows."""
    if isinstance(c, bool) or not isinstance(c, Real) or not isfinite(c) or c <= 0:
        raise ValueError("c must be a positive finite real number.")
    if len(fields) < 2 or not fields or not fields[0]:
        raise ValueError("fields must contain at least two non-empty field profiles.")

    K = len(fields)
    M = len(fields[0])
    if any(len(field) != M for field in fields):
        raise ValueError("all field profiles must have the same number of locations.")

    rows: list[CsvRow] = []
    for i, j in combinations(range(K), 2):
        for m in range(M):
            phase_gap = principal_phase(float(c) * (fields[i][m] - fields[j][m]))
            rows.append((i, j, m, phase_gap))
    return rows


def _location_metadata(locations: Sequence[Location]) -> list[object]:
    if len(locations[0]) == 1:
        return [location[0] for location in locations]
    return [list(location) for location in locations]


def generate_case(
    model: str,
    dimension: int,
    K: int,
    M: int,
    seed: int,
    c: Real = 1.0,
    max_attempts: int = 10_000,
) -> tuple[list[CsvRow], dict[str, object]]:
    """Generate one valid, reproducible spatial-field case."""
    if model not in FIELD_MODELS:
        available = ", ".join(sorted(FIELD_MODELS))
        raise ValueError(f"Unknown field model '{model}'. Available: {available}.")
    if dimension not in (1, 2):
        raise ValueError("dimension must be 1 or 2.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer.")

    required_pairs = set(combinations(range(K), 2))
    for attempt in range(1, max_attempts + 1):
        rng = Random(seed + attempt - 1)
        locations = sensing_locations(dimension, M, rng)
        fields, parameters = FIELD_MODELS[model](K, locations, rng)
        rows = phase_gap_rows(fields, c)
        distinguishable_pairs = {
            (i, j) for i, j, _, phase_gap in rows if phase_gap != 0.0
        }
        if distinguishable_pairs != required_pairs:
            continue
        return rows, {
            "field_model": model,
            "spatial_dimension": dimension,
            "K": K,
            "M": M,
            "phase_scale_c": float(c),
            "generation_seed": seed,
            "generation_attempt": attempt,
            "sensor_locations": _location_metadata(locations),
            "field_parameters": parameters,
        }

    raise RuntimeError(
        f"Could not construct a valid {dimension}D {model} case "
        f"after {max_attempts} attempts."
    )
