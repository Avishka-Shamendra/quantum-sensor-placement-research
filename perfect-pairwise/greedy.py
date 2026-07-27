"""Greedy heuristic for perfect pairwise sensor placement.

This is Algorithm 2 from the paper.  It repeatedly chooses the location and
sensor-count upgrade with the highest number of newly covered function pairs
per additional sensor.
"""

from collections.abc import Mapping, Sequence


def greedy_perfect_pairwise_sensor_placement(
    F: int,
    M: int,
    Nijm: Mapping[tuple[int, int], Sequence[int]],
) -> tuple[int, list[int], list[list[int | None]]]:
    """Return the greedy perfect-pairwise sensor allocation.

    Function and location indices are zero-based.  ``Nijm[(i, j)][m]`` is the
    positive number of sensors needed to perfectly distinguish functions
    ``i`` and ``j`` at location ``m``.

    Args:
        F: Number of candidate functions, indexed from ``0`` to ``F - 1``.
        M: Number of candidate sensor locations, indexed from ``0`` to
            ``M - 1``.
        Nijm: A mapping for every pair ``(i, j)`` with ``0 <= i < j < F``.
            Each value is a positive-integer sequence of length ``M``.

    Returns:
        ``(total, allocation, pair_locations)``.  ``allocation[m]`` is the
        final sensor count at location ``m``.  ``pair_locations[i][j]`` gives
        the location selected when pair ``(i, j)`` was first covered; the
        matrix is symmetric and has ``None`` on its diagonal.

    Raises:
        ValueError: If the inputs do not satisfy the problem definition.

    Notes:
        This follows the paper's strict ``gain > best_gain`` rule.  Ties are
        consequently resolved by the first location and count encountered.
    """
    if isinstance(F, bool) or not isinstance(F, int) or F < 2:
        raise ValueError("F must be an integer of at least 2.")
    if isinstance(M, bool) or not isinstance(M, int) or M < 1:
        raise ValueError("M must be a positive integer.")
    if not isinstance(Nijm, Mapping):
        raise ValueError("Nijm must be a mapping keyed by function-pair tuples.")

    pairs = tuple((i, j) for i in range(F) for j in range(i + 1, F))
    if set(Nijm) != set(pairs):
        raise ValueError(
            "Nijm must contain exactly one (i, j) key with 0 <= i < j < F "
            "for every function pair."
        )

    for pair in pairs:
        thresholds = Nijm[pair]
        if len(thresholds) != M:
            raise ValueError("Every Nijm[(i, j)] sequence must have length M.")
        for threshold in thresholds:
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, int)
                or threshold <= 0
            ):
                raise ValueError("Each Nijm threshold must be a positive integer.")

    # Algorithm 2: precompute Counts_m and Cover_m(n) for every location.
    location_options: list[list[tuple[int, frozenset[tuple[int, int]]]]] = []
    for m in range(M):
        counts = sorted({0} | {Nijm[pair][m] for pair in pairs})
        options = []
        for count in counts:
            covered = frozenset(
                pair for pair in pairs if count >= Nijm[pair][m]
            )
            options.append((count, covered))
        location_options.append(options)

    covered_pairs: set[tuple[int, int]] = set()
    allocation = [0] * M
    pair_locations: list[list[int | None]] = [
        [None for _ in range(F)] for _ in range(F)
    ]

    while len(covered_pairs) != len(pairs):
        best_gain = 0.0
        best_location: int | None = None
        best_count: int | None = None
        best_pairs: set[tuple[int, int]] = set()

        for m, options in enumerate(location_options):
            for count, coverage in options:
                if count <= allocation[m]:
                    continue

                new_pairs = set(coverage - covered_pairs)
                if not new_pairs:
                    continue

                gain = len(new_pairs) / (count - allocation[m])
                if gain > best_gain:
                    best_gain = gain
                    best_location = m
                    best_count = count
                    best_pairs = new_pairs

        # Positive finite thresholds guarantee that every uncovered pair has
        # an available improving move.  Keep this guard for malformed inputs.
        if best_location is None or best_count is None:
            raise ValueError("No greedy upgrade can cover the remaining pairs.")

        for i, j in best_pairs:
            pair_locations[i][j] = best_location
            pair_locations[j][i] = best_location

        covered_pairs.update(best_pairs)
        allocation[best_location] = best_count

    return sum(allocation), allocation, pair_locations
