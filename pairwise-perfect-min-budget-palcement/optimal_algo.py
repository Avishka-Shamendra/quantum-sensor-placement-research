"""Exact sensor placement for perfect pairwise function discrimination."""


def optimal_perfect_pairwise_sensor_placement(
    K,
    M,
    Nijm,
):
    """Find the exact minimum sensor allocation.

    Args:
        K: Number of candidate functions.
        M: Number of candidate sensor locations.
        Nijm: Dictionary mapping each pair ``(i, j)``, where ``i < j``, to a
            list of M sensor thresholds. ``Nijm[(i, j)][m]`` is the number of
            sensors required at location ``m`` to distinguish the pair.

    Returns:
        A tuple ``(Nmin, sensor_allocation, mij)`` where:

        - ``Nmin`` is the minimum total number of sensors.
        - ``sensor_allocation[m]`` is the number placed at location ``m``.
        - ``mij[i][j]`` is the zero-based location index assigned to pair
          ``(i, j)``. The matrix is symmetric and has ``None`` on its diagonal.

    Raises:
        ValueError: If the inputs have inconsistent dimensions, contain an
            invalid sensor threshold.

    Notes:
        The worst-case time is O(M * 2^|P| * |P|), where
        |P| = K(K - 1)/2.
    """
    if not isinstance(K, int) or K < 2:
        raise ValueError("At least two candidate functions are required.")
    if not isinstance(M, int) or M < 1:
        raise ValueError("At least one candidate location is required.")

    pairs = tuple((i, j) for i in range(K) for j in range(i + 1, K))

    # Require one threshold list for every unordered function pair.
    if not isinstance(Nijm, dict) or set(Nijm) != set(pairs):
        raise ValueError(
            "Nijm must contain exactly one (i, j) key with i < j "
            "for every function pair."
        )

    for pair in pairs:
        if len(Nijm[pair]) != M:
            raise ValueError("Every Nijm pair must contain M thresholds.")

        for threshold in Nijm[pair]:
            if threshold <= 0 or int(threshold) != threshold:
                raise ValueError(
                    "Every Nijm threshold must be a positive integer."
                )

    all_pairs = frozenset(pairs)

    # Only threshold values can change which pairs a location covers.
    # Example for one location:
    # candidate_counts[m] = [0, 2, 3, 4]  # useful sensor choices
    # coverage[m] = {
    #     0: frozenset(),                         # covers no pairs
    #     2: frozenset({(0, 1)}),                 # covers one pair
    #     3: frozenset({(0, 1), (1, 2)}),         # covers two pairs
    #     4: frozenset({(0, 1), (0, 2), (1, 2)}), # covers all pairs
    # }
    candidate_counts = []
    coverage = []

    for m in range(M):
        counts = {Nijm[pair][m] for pair in pairs}
        counts.add(0)
        counts = sorted(counts)
        candidate_counts.append(counts)

        # Map each useful count to the pairs it covers at this location.
        coverage.append(
            {
                n: frozenset(
                    pair
                    for pair in pairs
                    if n >= Nijm[pair][m]
                )
                for n in counts
            }
        )

    # dp[S] stores the minimum cost of covering pair subset S using all
    # locations processed so far. Missing subsets have infinite cost.
    # The initial layer covers no pairs at zero cost.
    dp = {frozenset(): 0}

    # Save each layer's chosen transitions for final backtracking.
    parent_layers = []

    for m in range(M):
        # At the start, dp represents DP[m, S]. Build DP[m + 1, S] in
        # next_dp by considering every useful sensor count at location m.
        next_dp = {}
        parents = {}

        for covered_pairs, current_cost in dp.items():
            for n in candidate_counts[m]:
                new_pairs = covered_pairs | coverage[m][n]
                new_cost = current_cost + n

                # Keep the cheapest allocation for each covered-pair subset.
                if (
                    new_pairs not in next_dp
                    or new_cost < next_dp[new_pairs]
                ):
                    next_dp[new_pairs] = new_cost
                    parents[new_pairs] = (covered_pairs, n)

        # Keep the newly completed layer and discard the older cost layer.
        dp = next_dp
        parent_layers.append(parents)

    Nmin = dp[all_pairs]

    # Backtrack to recover the sensor count at each location.
    sensor_allocation = [0] * M
    current_pairs = all_pairs

    for m in range(M - 1, -1, -1):
        previous_pairs, n = parent_layers[m][current_pairs]
        sensor_allocation[m] = n
        current_pairs = previous_pairs

    # Assign each pair to the first allocated location that covers it.
    mij = [[None for _ in range(K)] for _ in range(K)]

    for i, j in pairs:
        selected_location = next(
            m
            for m in range(M)
            if sensor_allocation[m] >= Nijm[(i, j)][m]
        )
        mij[i][j] = selected_location
        mij[j][i] = selected_location

    return Nmin, sensor_allocation, mij
