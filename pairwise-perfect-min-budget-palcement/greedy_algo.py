"""Greedy sensor placement for perfect pairwise function discrimination."""


def greedy_perfect_pairwise_sensor_placement(
    K,
    M,
    Nijm,
):
    """Find a sensor allocation using a greedy gain-per-sensor strategy.

    Args:
        K: Number of candidate functions.
        M: Number of candidate sensor locations.
        Nijm: Dictionary mapping each pair ``(i, j)``, where ``i < j``, to a
            list of M sensor thresholds. ``Nijm[(i, j)][m]`` is the number of
            sensors required at location ``m`` to distinguish the pair.

    Returns:
        A tuple ``(total_sensors, sensor_allocation, mij)`` where:

        - ``total_sensors`` is the total number selected by the greedy method.
          It is not guaranteed to equal the exact minimum.
        - ``sensor_allocation[m]`` is the number placed at location ``m``.
        - ``mij[i][j]`` is the zero-based location index assigned to pair
          ``(i, j)``. The matrix is symmetric and has ``None`` on its diagonal.

    Raises:
        ValueError: If the inputs have inconsistent dimensions, contain an
            invalid sensor threshold.

    Notes:
        Each iteration chooses the sensor-count increase with maximum:

            newly covered function pairs / extra sensors added
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

    all_pairs = set(pairs)

    # Precompute useful counts and the pairs covered by each count.
    # Example for one location:
    # candidate_counts[m] = [0, 2, 3, 4]  # useful sensor choices
    # coverage[m] = {
    #     0: set(),                         # covers no pairs
    #     2: {(0, 1)},                      # covers one pair
    #     3: {(0, 1), (1, 2)},              # covers two pairs
    #     4: {(0, 1), (0, 2), (1, 2)},      # covers all pairs
    # }
    candidate_counts = []
    coverage = []

    for m in range(M):
        counts = {Nijm[pair][m] for pair in pairs}
        counts.add(0)
        counts = sorted(counts)
        candidate_counts.append(counts)

        coverage.append(
            {
                n: {
                    pair
                    for pair in pairs
                    if n >= Nijm[pair][m]
                }
                for n in counts
            }
        )

    covered_pairs = set()
    sensor_allocation = [0] * M
    mij = [[None for _ in range(K)] for _ in range(K)]

    # Select the highest-gain increase until every pair is covered.
    while covered_pairs != all_pairs:
        best_gain = 0
        best_location = None
        best_count = None
        best_new_pairs = set()

        for m in range(M):
            for n in candidate_counts[m]:
                # Only consider increases to the current allocation.
                if n <= sensor_allocation[m]:
                    continue

                new_pairs = coverage[m][n] - covered_pairs
                extra_sensors = n - sensor_allocation[m]

                if not new_pairs:
                    continue

                gain = len(new_pairs) / extra_sensors

                # Keep the first location and count when gains are equal.
                if gain > best_gain:
                    best_gain = gain
                    best_location = m
                    best_count = n
                    best_new_pairs = new_pairs

        if best_location is None:
            raise RuntimeError("No valid sensor-count increase was found.")

        # Assign newly covered pairs to the selected location.
        for i, j in best_new_pairs:
            mij[i][j] = best_location
            mij[j][i] = best_location

        covered_pairs.update(best_new_pairs)
        sensor_allocation[best_location] = best_count

    total_sensors = sum(sensor_allocation)
    return total_sensors, sensor_allocation, mij
