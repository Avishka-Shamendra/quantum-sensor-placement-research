"""Synthetic spatial-field models for the zero-error experiments.

Each field-family function creates ``K`` candidate spatial fields sampled on
the supplied locations.  The shared conversion then applies the paper's
definition

    r_ijm = Arg(exp(i c (f_i(x_m) - f_j(x_m)))).

Only canonical pairs ``i < j`` are emitted. The reverse phase is obtained as
``r_jim = principal_phase(-r_ijm)``, so antisymmetry is maintained modulo
``2 pi`` without storing redundant CSV rows.

The available families are deliberately simple and interpretable:

``gaussian``
    Localized Gaussian sources with different centres, amplitudes, and widths.
``sine``
    Single-frequency sinusoidal fields with different amplitudes, frequencies,
    and phases.
``fourier``
    Smooth fields formed from the first two Fourier harmonics.
``polynomial``
    Smooth low-order background/gradient fields on the sensing interval.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations
from math import cos, exp, isfinite, pi, sin
from numbers import Real
from random import Random
from typing import Union


FieldValues = list[list[float]]
FieldParameters = list[dict[str, Union[int, float]]]
FieldGeneration = tuple[FieldValues, FieldParameters]
CsvRow = tuple[int, int, int, float]
FieldFactory = Callable[[int, Sequence[float], Random], FieldGeneration]


def sensing_locations(M: int) -> list[float]:
    """Return ``M`` evenly spaced sensing locations on the interval [0, 1]."""
    if isinstance(M, bool) or not isinstance(M, int) or M < 2:
        raise ValueError("M must be an integer of at least 2.")
    return [m / (M - 1) for m in range(M)]


def _validate_field_request(K: int, locations: Sequence[float]) -> None:
    if isinstance(K, bool) or not isinstance(K, int) or K < 2:
        raise ValueError("K must be an integer of at least 2.")
    if not locations:
        raise ValueError("locations must not be empty.")


def generate_gaussian_fields(
    K: int, locations: Sequence[float], rng: Random
) -> FieldGeneration:
    """Generate pure localized Gaussian fields.

    ``f_i(x) = A_i exp(-(x-mu_i)^2 / (2 sigma_i^2))``.
    """
    _validate_field_request(K, locations)
    centres = [
        (i + 0.5) / K + rng.uniform(-0.055, 0.055)
        for i in range(K)
    ]
    fields: FieldValues = []
    parameters: FieldParameters = []
    for i in range(K):
        amplitude = rng.uniform(1.10, 1.40)
        width = rng.uniform(0.28, 0.36)
        parameters.append({
            "field_index": i,
            "amplitude": amplitude,
            "centre": centres[i],
            "width": width,
        })
        fields.append([
            amplitude * exp(-((x - centres[i]) ** 2) / (2 * width**2))
            for x in locations
        ])
    return fields, parameters


def generate_sine_fields(
    K: int, locations: Sequence[float], rng: Random
) -> FieldGeneration:
    """Generate single-frequency sinusoidal fields.

    ``f_i(x) = A_i sin(2 pi q_i x + phi_i)`` where ``q_i`` is 1, 2, or 3.
    """
    _validate_field_request(K, locations)
    fields: FieldValues = []
    parameters: FieldParameters = []
    for i in range(K):
        amplitude = rng.uniform(0.55, 1.20)
        frequency = rng.randint(1, 3)
        phase = rng.uniform(-pi, pi)
        parameters.append({
            "field_index": i,
            "amplitude": amplitude,
            "frequency": frequency,
            "phase": phase,
        })
        fields.append([
            amplitude * sin(2 * pi * frequency * x + phase)
            for x in locations
        ])
    return fields, parameters


def generate_fourier_fields(
    K: int, locations: Sequence[float], rng: Random
) -> FieldGeneration:
    """Generate smooth fields with two spatial Fourier harmonics.

    ``f_i(x) = a0 + a1 sin(2 pi x) + b1 cos(2 pi x)
              + a2 sin(4 pi x) + b2 cos(4 pi x)``.
    """
    _validate_field_request(K, locations)
    fields: FieldValues = []
    parameters: FieldParameters = []
    for i in range(K):
        a0 = rng.uniform(-0.18, 0.18)
        a1 = rng.uniform(-0.48, 0.48)
        b1 = rng.uniform(-0.48, 0.48)
        a2 = rng.uniform(-0.20, 0.20)
        b2 = rng.uniform(-0.20, 0.20)
        parameters.append({
            "field_index": i,
            "a0": a0,
            "a1": a1,
            "b1": b1,
            "a2": a2,
            "b2": b2,
        })
        fields.append([
            a0
            + a1 * sin(2 * pi * x)
            + b1 * cos(2 * pi * x)
            + a2 * sin(4 * pi * x)
            + b2 * cos(4 * pi * x)
            for x in locations
        ])
    return fields, parameters


def generate_polynomial_fields(
    K: int, locations: Sequence[float], rng: Random
) -> FieldGeneration:
    """Generate smooth cubic background/gradient fields.

    ``f_i(x) = a0 + a1 u + a2 u^2 + a3 u^3``, with ``u = x - 1/2``.
    """
    _validate_field_request(K, locations)
    fields: FieldValues = []
    parameters: FieldParameters = []
    for i in range(K):
        a0 = rng.uniform(-0.20, 0.20)
        a1 = rng.uniform(-1.00, 1.00)
        a2 = rng.uniform(-0.80, 0.80)
        a3 = rng.uniform(-0.40, 0.40)
        parameters.append({
            "field_index": i,
            "a0": a0,
            "a1": a1,
            "a2": a2,
            "a3": a3,
        })
        fields.append([
            a0 + a1 * (x - 0.5) + a2 * (x - 0.5) ** 2 + a3 * (x - 0.5) ** 3
            for x in locations
        ])
    return fields, parameters


FIELD_MODELS: dict[str, FieldFactory] = {
    "gaussian": generate_gaussian_fields,
    "sine": generate_sine_fields,
    "fourier": generate_fourier_fields,
    "polynomial": generate_polynomial_fields,
}

FIELD_FORMULAS = {
    "gaussian": "f_i(x) = A_i exp(-(x-mu_i)^2 / (2 sigma_i^2))",
    "sine": "f_i(x) = A_i sin(2 pi q_i x + phi_i)",
    "fourier": (
        "f_i(x) = a0_i + a1_i sin(2 pi x) + b1_i cos(2 pi x) "
        "+ a2_i sin(4 pi x) + b2_i cos(4 pi x)"
    ),
    "polynomial": "f_i(x) = a0_i + a1_i u + a2_i u^2 + a3_i u^3, u = x - 1/2",
}


def principal_phase(value: Real) -> float:
    """Wrap a finite phase to the interval ``(-pi, pi]``."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("phase value must be a finite real number.")
    wrapped = (float(value) + pi) % (2 * pi) - pi
    return pi if wrapped <= -pi else wrapped


def phase_gap_rows(fields: FieldValues, c: Real) -> list[CsvRow]:
    """Convert sampled field values into canonical Section III phase rows."""
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
            phase_gap = principal_phase(
                float(c) * (fields[i][m] - fields[j][m])
            )
            rows.append((i, j, m, phase_gap))
    return rows


def generate_case(
    model: str,
    K: int,
    M: int,
    seed: int,
    c: Real = 1.0,
    max_attempts: int = 10_000,
) -> tuple[list[CsvRow], dict[str, object]]:
    """Generate one valid deterministic case from a selected field model."""
    if model not in FIELD_MODELS:
        available = ", ".join(sorted(FIELD_MODELS))
        raise ValueError(f"Unknown field model '{model}'. Available: {available}.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer.")

    locations = sensing_locations(M)
    for attempt in range(1, max_attempts + 1):
        fields, parameters = FIELD_MODELS[model](
            K, locations, Random(seed + attempt)
        )
        rows = phase_gap_rows(fields, c)
        distinguishable_pairs = {
            (i, j)
            for i, j, _, phase_gap in rows
            if phase_gap != 0.0
        }
        required_pairs = set(combinations(range(K), 2))
        if distinguishable_pairs != required_pairs:
            continue
        return rows, {
            "field_model": model,
            "K": K,
            "M": M,
            "phase_scale_c": float(c),
            "generation_seed": seed,
            "generation_attempt": attempt,
            "sensor_locations": locations,
            "field_parameters": parameters,
        }

    raise RuntimeError(
        f"Could not construct a valid {model} case after {max_attempts} attempts."
    )
