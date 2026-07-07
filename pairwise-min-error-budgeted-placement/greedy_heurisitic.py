"""Greedy minimum-error budgeted pairwise sensor placement."""

from itertools import combinations


def greedy_min_error_pairwise_sensor_placement(K, M, N, Pijmn):
    """Greedily place sensors to minimize adaptive pairwise error.

    Args:
        K: Number of candidate functions.
        M: Number of candidate sensor locations.
        N: Total number of sensors to place.
        Pijmn: Pairwise error probabilities. The preferred format is
            ``Pijmn[(i, j, m, n)] = error`` with ``i < j``. The value is the
            probability of discarding the active function when comparing
            ``f_i`` and ``f_j`` at location ``m`` with ``n`` sensors.

            The nested formats ``Pijmn[(i, j, m)][n]`` and
            ``Pijmn[(i, j)][m][n]`` are also accepted.

    Returns:
        A tuple ``(final_error, sensor_allocation, mij, tree)`` where:

        - ``final_error`` is the average final misclassification probability
          of the greedy allocation and greedy adaptive tree.
        - ``sensor_allocation[m]`` is the number of sensors placed at location
          ``m``.
        - ``mij[i][j]`` is the zero-based best test location for pair
          ``(i, j)`` under the returned allocation. The matrix is symmetric and
          has ``None`` on its diagonal.
        - ``tree`` is a nested dictionary. Internal nodes have
          ``test_pair=(i, j)`` and branches ``discard_i`` and ``discard_j``.
          Leaf nodes have ``declare=k``.

    Raises:
        ValueError: If the problem dimensions are invalid, no feasible greedy
            trial can evaluate every pair, or an error probability is outside
            ``[0, 0.5]``.
    """
    _validate_dimensions(K, M, N)

    pairs = tuple((i, j) for i in range(K) for j in range(i + 1, K))
    allocation = [0] * M

    for _ in range(N):
        best_trial_error = float("inf")
        best_trial_location = None

        # Try placing the next sensor at each location.
        for r in range(M):
            trial_allocation = allocation[:]
            trial_allocation[r] += 1

            # Compute e_ij values induced by this temporary allocation.
            effective_errors, _ = _effective_errors_for_allocation(
                K, M, pairs, trial_allocation, Pijmn
            )

            if effective_errors is None:
                continue

            # Build the greedy tree using the smallest e_ij at each node.
            tree = _build_greedy_tree(tuple(range(K)), effective_errors)
            trial_error = 1.0 - _evaluate_tree_success(
                K, tree, effective_errors
            )

            # Permanently choose the trial location with lowest final error.
            if trial_error < best_trial_error:
                best_trial_error = trial_error
                best_trial_location = r

        if best_trial_location is None:
            raise ValueError(
                "No feasible greedy trial could define an effective error for "
                "every function pair."
            )

        allocation[best_trial_location] += 1

    # Recompute final pair locations and final greedy tree after all N sensors.
    effective_errors, mij = _effective_errors_for_allocation(
        K, M, pairs, allocation, Pijmn
    )

    if effective_errors is None:
        raise ValueError(
            "The final greedy allocation cannot evaluate every function pair."
        )

    tree = _build_greedy_tree(tuple(range(K)), effective_errors)
    final_error = 1.0 - _evaluate_tree_success(K, tree, effective_errors)

    return final_error, allocation, mij, tree


def _validate_dimensions(K, M, N):
    if not isinstance(K, int) or K < 2:
        raise ValueError("At least two candidate functions are required.")
    if not isinstance(M, int) or M < 1:
        raise ValueError("At least one candidate location is required.")
    if not isinstance(N, int) or N < 0:
        raise ValueError("The sensor budget N must be a nonnegative integer.")


def _effective_errors_for_allocation(K, M, pairs, allocation, Pijmn):
    effective_errors = {}
    mij = [[None for _ in range(K)] for _ in range(K)]

    for i, j in pairs:
        best_error = None
        best_location = None

        for m in range(M):
            # A pair can use only the sensors allocated to this location.
            error = _lookup_error(Pijmn, i, j, m, allocation[m])

            if error is None:
                continue

            _validate_error_probability(error, i, j, m, allocation[m])

            # Effective pairwise error is the best location-specific error.
            if best_error is None or error < best_error:
                best_error = error
                best_location = m

        if best_error is None:
            return None, None

        effective_errors[(i, j)] = best_error
        mij[i][j] = best_location
        mij[j][i] = best_location

    return effective_errors, mij


def _build_greedy_tree(remaining, effective_errors):
    if len(remaining) == 1:
        declared = remaining[0]
        return {
            "remaining": list(remaining),
            "declare": declared,
        }

    # Greedy choice: test the remaining pair with smallest effective error.
    i, j = min(
        combinations(remaining, 2),
        key=lambda pair: effective_errors[
            (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
        ],
    )

    remaining_without_i = tuple(k for k in remaining if k != i)
    remaining_without_j = tuple(k for k in remaining if k != j)

    return {
        "remaining": list(remaining),
        "test_pair": (i, j),
        "discard_i": _build_greedy_tree(
            remaining_without_i, effective_errors
        ),
        "discard_j": _build_greedy_tree(
            remaining_without_j, effective_errors
        ),
    }


def _evaluate_tree_success(K, tree, effective_errors):
    root_weights = tuple(1.0 / K for _ in range(K))
    return _evaluate_tree_success_from_weights(
        K, tree, effective_errors, root_weights
    )


def _evaluate_tree_success_from_weights(K, tree, effective_errors, weights):
    if "declare" in tree:
        # A leaf succeeds only for the probability mass of its declaration.
        return weights[tree["declare"]]

    i, j = tree["test_pair"]
    pair = (i, j) if i < j else (j, i)
    error = effective_errors[pair]

    # Propagate each possible active function down both outcome branches.
    weights_after_discard_i = _branch_weights(
        K, tuple(tree["discard_i"]["remaining"]), weights,
        tested_pair=(i, j), discarded=i, error=error
    )
    weights_after_discard_j = _branch_weights(
        K, tuple(tree["discard_j"]["remaining"]), weights,
        tested_pair=(i, j), discarded=j, error=error
    )

    return (
        _evaluate_tree_success_from_weights(
            K, tree["discard_i"], effective_errors, weights_after_discard_i
        )
        + _evaluate_tree_success_from_weights(
            K, tree["discard_j"], effective_errors, weights_after_discard_j
        )
    )


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
