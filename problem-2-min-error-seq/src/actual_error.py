"""Decision-tree error from the closed-form outcome probabilities in Appendix D."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, cos, isclose, pi, sin

from input_data import PhaseGaps, pair_error_probability, principal_phase


@dataclass(frozen=True)
class NodeProbabilities:
    """Two outcome probabilities for one true field at one tree node."""

    keep_i: float
    keep_j: float


def actual_node_probabilities(
    pair: tuple[int, int],
    location: int,
    sensor_count: int,
    true_field: int,
    phase_gaps: PhaseGaps,
) -> NodeProbabilities:
    """Return the Appendix D outcome probabilities for one pairwise test.

    ``keep_i`` is ``Q_ijmn(k -> i)``: the probability of the outcome favoring
    field ``i``. In an elimination tree, this outcome discards field ``j``.
    """
    if sensor_count < 0:
        raise ValueError("sensor_count must be nonnegative")

    i, j = pair
    pair_error = pair_error_probability(
        _signed_phase_difference(i, j, location, phase_gaps), sensor_count
    )
    if true_field == i:
        keep_i = 1.0 - pair_error
    elif true_field == j:
        keep_i = pair_error
    else:
        keep_i = off_pair_probability(
            i, j, true_field, location, sensor_count, phase_gaps
        )

    keep_i = min(1.0, max(0.0, keep_i))
    return NodeProbabilities(keep_i=keep_i, keep_j=1.0 - keep_i)


def off_pair_probability(
    i: int,
    j: int,
    k: int,
    location: int,
    sensor_count: int,
    phase_gaps: PhaseGaps,
) -> float:
    """Evaluate ``Q_ijmn(k -> i)`` using Appendix D, equations (38)-(47)."""
    if k in (i, j):
        raise ValueError("k must be outside the tested pair")
    if sensor_count < 0:
        raise ValueError("sensor_count must be nonnegative")

    rij = _signed_phase_difference(i, j, location, phase_gaps)
    rik = _signed_phase_difference(i, k, location, phase_gaps)
    rjk = _signed_phase_difference(j, k, location, phase_gaps)
    gap = abs(rij)

    # Appendix D, equation (38).
    if sensor_count == 0 or gap == 0.0:
        return 0.5

    threshold = ceil(pi / gap)

    # Appendix D, equation (41).
    if sensor_count < threshold:
        numerator = cos(sensor_count * rik / 2.0) ** 2 - cos(
            sensor_count * rjk / 2.0
        ) ** 2
        denominator = 2.0 * sin(sensor_count * gap / 2.0)
        return 0.5 + numerator / denominator

    # For n >= N_ijm, Appendix D uses only s = N_ijm sensors.
    s = threshold
    if isclose(s * gap, pi, rel_tol=0.0, abs_tol=1e-12):
        # Equation (43).
        overlap_ik = cos(s * rik / 2.0)
        overlap_jk = cos(s * rjk / 2.0)
    elif s % 2 == 0:
        # Equations (44)-(45).
        tested_cosine = cos(s * gap / 2.0)
        denominator = 1.0 - tested_cosine
        overlap_ik = (cos(s * rik / 2.0) - tested_cosine) / denominator
        overlap_jk = (cos(s * rjk / 2.0) - tested_cosine) / denominator
    else:
        # Equations (46)-(47).
        single_cosine = cos(gap / 2.0)
        tested_cosine = cos(s * gap / 2.0)
        denominator = single_cosine - tested_cosine
        overlap_ik = (
            single_cosine * cos(s * rik / 2.0)
            - tested_cosine * cos(rik / 2.0)
        ) / denominator
        overlap_jk = (
            single_cosine * cos(s * rjk / 2.0)
            - tested_cosine * cos(rjk / 2.0)
        ) / denominator

    return 0.5 + 0.5 * (overlap_ik**2 - overlap_jk**2)


def actual_tree_error(
    tree: Mapping[str, object] | object,
    K: int,
    phase_gaps: PhaseGaps,
    allocation: Sequence[int],
) -> float:
    """Evaluate a returned tree with the Appendix D branch probabilities."""
    if K < 2:
        raise ValueError("K must be at least 2")
    if len(allocation) == 0:
        raise ValueError("allocation must contain at least one location")

    success = sum(
        _leaf_success_probability(tree, field, phase_gaps, allocation)
        for field in range(K)
    )
    return 1.0 - success / K


def _signed_phase_difference(
    first: int,
    second: int,
    location: int,
    phase_gaps: PhaseGaps,
) -> float:
    if first == second:
        return 0.0
    if first < second:
        return float(phase_gaps[(first, second)][location])
    return principal_phase(-float(phase_gaps[(second, first)][location]))


def _leaf_success_probability(
    tree: Mapping[str, object] | object,
    true_field: int,
    phase_gaps: PhaseGaps,
    allocation: Sequence[int],
) -> float:
    declared = _declared_field(tree)
    if declared is not None:
        return 1.0 if declared == true_field else 0.0

    pair = _tree_pair(tree)
    location = _tree_location(tree)
    probabilities = actual_node_probabilities(
        pair,
        location,
        int(allocation[location]),
        true_field,
        phase_gaps,
    )
    return (
        probabilities.keep_j
        * _leaf_success_probability(
            _tree_child(tree, "discard_i"), true_field, phase_gaps, allocation
        )
        + probabilities.keep_i
        * _leaf_success_probability(
            _tree_child(tree, "discard_j"), true_field, phase_gaps, allocation
        )
    )


def _declared_field(tree: Mapping[str, object] | object) -> int | None:
    if isinstance(tree, Mapping):
        value = tree.get("declared_field")
    else:
        value = getattr(tree, "declared_field", None)
    return None if value is None else int(value)


def _tree_pair(tree: Mapping[str, object] | object) -> tuple[int, int]:
    value = tree["pair"] if isinstance(tree, Mapping) else getattr(tree, "pair")
    i, j = value
    return int(i), int(j)


def _tree_location(tree: Mapping[str, object] | object) -> int:
    value = tree["location"] if isinstance(tree, Mapping) else getattr(tree, "location")
    return int(value)


def _tree_child(
    tree: Mapping[str, object] | object,
    name: str,
) -> Mapping[str, object] | object:
    return tree[name] if isinstance(tree, Mapping) else getattr(tree, name)
