"""Exact minimum-error budgeted pairwise sensor placement."""

from functools import lru_cache
from itertools import combinations


def optimal_min_error_pairwise_sensor_placement(K, M, N, Pijmn):
    """Find the exact minimum-error budgeted sensor allocation.

    Args:
        K: Number of candidate functions.
        M: Number of candidate sensor locations.
        N: Total number of sensors to place.
        Pijmn: Pairwise error probabilities. The most direct format is a
            dictionary with keys ``(i, j, m, n)``, where ``i < j``. The value is
            the probability of discarding the active function when comparing
            ``f_i`` and ``f_j`` at location ``m`` with ``n`` sensors.

            For convenience, this function also accepts either
            ``Pijmn[(i, j, m)][n]`` or ``Pijmn[(i, j)][m][n]`` when those nested
            containers are easier to build.

    Returns:
        A tuple ``(best_error, sensor_allocation, mij, tree)`` where:

        - ``best_error`` is the minimum average final misclassification
          probability.
        - ``sensor_allocation[m]`` is the number of sensors placed at location
          ``m``.
        - ``mij[i][j]`` is the zero-based best test location for pair
          ``(i, j)`` under the returned allocation. The matrix is symmetric and
          has ``None`` on its diagonal.
        - ``tree`` is a nested dictionary. Internal nodes have
          ``test_pair=(i, j)`` and two branches, ``discard_i`` and
          ``discard_j``. Leaf nodes have ``declare=k``.

    Raises:
        ValueError: If the problem dimensions are invalid, no feasible
            allocation can evaluate every pair, or an error probability is
            outside ``[0, 0.5]``.

    Notes:
        This is an exact exhaustive-budget algorithm. For each allocation, it
        optimizes the adaptive elimination tree exactly by recursively carrying
        the probability mass for each still-possible active function. That is
        equivalent to enumerating adaptive trees, but avoids storing all full
        trees at once.
    """
    _validate_dimensions(K, M, N)

    pairs = tuple((i, j) for i in range(K) for j in range(i + 1, K))
    best_error = float("inf")
    best_allocation = None
    best_mij = None
    best_tree = None

    # Try every feasible way to spend exactly N sensors across M locations.
    for allocation in _sensor_allocations(M, N):
        # Convert location-specific Pijmn values into one best error per pair.
        effective_errors, mij = _effective_errors_for_allocation(
            K, M, pairs, allocation, Pijmn
        )

        if effective_errors is None:
            continue

        # For this allocation, find the best adaptive elimination tree.
        tree_success, tree = _optimal_tree_for_effective_errors(
            K, effective_errors
        )
        error = 1.0 - tree_success

        # Keep the allocation/tree pair with the smallest average final error.
        if error < best_error:
            best_error = error
            best_allocation = list(allocation)
            best_mij = mij
            best_tree = tree

    if best_allocation is None:
        raise ValueError(
            "No feasible allocation could define an effective error for every "
            "function pair."
        )

    return best_error, best_allocation, best_mij, best_tree


def _validate_dimensions(K, M, N):
    if not isinstance(K, int) or K < 2:
        raise ValueError("At least two candidate functions are required.")
    if not isinstance(M, int) or M < 1:
        raise ValueError("At least one candidate location is required.")
    if not isinstance(N, int) or N < 0:
        raise ValueError("The sensor budget N must be a nonnegative integer.")


def _sensor_allocations(M, N):
    if M == 1:
        yield (N,)
        return

    for first_count in range(N + 1):
        for suffix in _sensor_allocations(M - 1, N - first_count):
            yield (first_count,) + suffix


def _effective_errors_for_allocation(K, M, pairs, allocation, Pijmn):
    effective_errors = {}
    mij = [[None for _ in range(K)] for _ in range(K)]

    for i, j in pairs:
        best_error = None
        best_location = None

        for m in range(M):
            # The pair can only use the number of sensors placed at location m.
            error = _lookup_error(Pijmn, i, j, m, allocation[m])

            if error is None:
                continue

            _validate_error_probability(error, i, j, m, allocation[m])

            # Pick the location giving the smallest available pairwise error.
            if best_error is None or error < best_error:
                best_error = error
                best_location = m

        if best_error is None:
            return None, None

        effective_errors[(i, j)] = best_error
        mij[i][j] = best_location
        mij[j][i] = best_location

    return effective_errors, mij


def _lookup_error(Pijmn, i, j, m, n):
    pair = (i, j) if i < j else (j, i)
    reverse_pair = (pair[1], pair[0])

    # First try the exact flat dictionary format.
    flat_keys = (
        (pair[0], pair[1], m, n),
        (reverse_pair[0], reverse_pair[1], m, n),
    )

    for key in flat_keys:
        if key in Pijmn:
            return Pijmn[key]

    # If exact n is missing, use the largest explicitly known count <= n.
    floor_error = _lookup_flat_floor_error(Pijmn, pair, reverse_pair, m, n)
    if floor_error is not None:
        return floor_error

    location_keys = (
        (pair[0], pair[1], m),
        (reverse_pair[0], reverse_pair[1], m),
    )

    for key in location_keys:
        if key in Pijmn:
            return _lookup_count_error(Pijmn[key], n)

    for key in (pair, reverse_pair):
        if key in Pijmn:
            location_errors = _index_or_key(Pijmn[key], m)

            if location_errors is None:
                continue

            return _lookup_count_error(location_errors, n)

    # Missing pair/location data means the test is no better than random.
    return 0.5


def _lookup_flat_floor_error(Pijmn, pair, reverse_pair, m, n):
    best_count = None
    best_error = None

    for key, error in Pijmn.items():
        if not isinstance(key, tuple) or len(key) != 4:
            continue

        key_i, key_j, key_m, key_n = key
        key_pair = (key_i, key_j)

        if key_m != m or key_n > n:
            continue

        if key_pair != pair and key_pair != reverse_pair:
            continue

        if best_count is None or key_n > best_count:
            best_count = key_n
            best_error = error

    return best_error


def _lookup_count_error(values, n):
    if isinstance(values, dict):
        if n in values:
            return values[n]

        useful_counts = [
            count
            for count in values
            if isinstance(count, int) and count <= n
        ]

        if useful_counts:
            return values[max(useful_counts)]

        return 0.5

    try:
        start = min(n, len(values) - 1)
    except TypeError:
        return 0.5

    for count in range(start, -1, -1):
        error = values[count]

        if error is not None:
            return error

    return 0.5


def _index_or_key(values, key):
    if isinstance(values, dict):
        return values.get(key)

    try:
        return values[key]
    except (IndexError, KeyError, TypeError):
        return None


def _validate_error_probability(error, i, j, m, n):
    if not isinstance(error, (int, float)):
        raise ValueError(
            f"Pijmn[({i}, {j}, {m}, {n})] must be a numeric probability."
        )

    if error < 0 or error > 0.5:
        raise ValueError(
            f"Pijmn[({i}, {j}, {m}, {n})] must be between 0 and 0.5."
        )


def _optimal_tree_for_effective_errors(K, effective_errors):
    root_remaining = tuple(range(K))
    root_weights = tuple(1.0 / K for _ in range(K))

    @lru_cache(maxsize=None)
    def solve(remaining, weights):
        if len(remaining) == 1:
            # A leaf is correct exactly when its declared function is active.
            declared = remaining[0]
            return weights[declared], {
                "remaining": list(remaining),
                "declare": declared,
            }

        best_success = -1.0
        best_tree = None

        # Test every pair still alive at this node and keep the best branch tree.
        for i, j in combinations(remaining, 2):
            pair = (i, j) if i < j else (j, i)
            error = effective_errors[pair]

            remaining_without_i = tuple(k for k in remaining if k != i)
            remaining_without_j = tuple(k for k in remaining if k != j)

            # Branch probabilities depend on whether the active function is
            # tested; untested active functions choose either branch randomly.
            weights_after_discard_i = _branch_weights(
                K, remaining_without_i, weights, tested_pair=(i, j),
                discarded=i, error=error
            )
            weights_after_discard_j = _branch_weights(
                K, remaining_without_j, weights, tested_pair=(i, j),
                discarded=j, error=error
            )

            success_i, tree_i = solve(
                remaining_without_i, weights_after_discard_i
            )
            success_j, tree_j = solve(
                remaining_without_j, weights_after_discard_j
            )
            success = success_i + success_j

            # Maximizing success is the same as minimizing final error.
            if success > best_success:
                best_success = success
                best_tree = {
                    "remaining": list(remaining),
                    "test_pair": (i, j),
                    "discard_i": tree_i,
                    "discard_j": tree_j,
                }

        return best_success, best_tree

    return solve(root_remaining, root_weights)


def _branch_weights(K, remaining, weights, tested_pair, discarded, error):
    i, j = tested_pair
    next_weights = [0.0] * K

    for active in remaining:
        # If active is tested, error means discarding that active function.
        if active == i or active == j:
            branch_probability = error if active == discarded else 1.0 - error
        else:
            branch_probability = 0.5

        next_weights[active] = weights[active] * branch_probability

    return tuple(next_weights)
