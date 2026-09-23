"""Run matched actual-error comparisons with resumable per-method records."""

from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import fmean
from time import perf_counter
from zlib import crc32

from exhaustive_search import exhaustive_search
from greedy import greedy
from heuristic import heuristic
from input_data import load_phase_input
from random_baseline import random_baseline
from strategy import SearchTimeout, tree_to_dict
from uniform import uniform


PROJECT = Path(__file__).resolve().parents[1]
TEST_CASES_PER_DIMENSION = {"compact": 90, "extended": 180}
SOLVERS = {
    "Exhaustive": exhaustive_search,
    "Greedy": greedy,
    "Heuristic": heuristic,
    "Uniform": uniform,
    "Random": random_baseline,
}
COMPACT_METHODS = ("Exhaustive", "Greedy", "Heuristic", "Uniform", "Random")
EXTENDED_METHODS = ("Greedy", "Heuristic", "Uniform", "Random")


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_cases(project, study, mode, exact_k_max):
    if study not in ("compact", "extended") or mode not in ("test", "full"):
        raise ValueError("Choose compact/extended and test/full")
    if exact_k_max < 2:
        raise ValueError("exact_k_max must be >= 2")
    test_case_count = TEST_CASES_PER_DIMENSION[study]
    entries = []
    for dimension in (1, 2):
        dataset = (
            f"compact_benchmark_{dimension}d"
            if study == "compact"
            else f"benchmark_{dimension}d"
        )
        directory = project / "inputs" / dataset
        manifest = json.loads((directory / "manifest.json").read_text())
        eligible = [
            case
            for case in manifest["cases"]
            if study != "compact" or case["K"] <= exact_k_max
        ]
        if mode == "test":
            if study == "compact":
                key_fields = ("field_model", "K", "M")
                cells = sorted({
                    tuple(case[field] for field in key_fields)
                    for case in eligible
                })
                quotient, remainder = divmod(test_case_count, len(cells))
                selected = []
                for cell_index, cell in enumerate(cells):
                    quota = quotient + int(cell_index < remainder)
                    cell_cases = [
                        case
                        for case in eligible
                        if tuple(case[field] for field in key_fields) == cell
                    ]
                    if len(cell_cases) < quota:
                        raise ValueError(
                            f"{dataset} needs at least {quota} cases in cell {cell}"
                        )
                    selected.extend(cell_cases[:quota])
            else:
                pairs = sorted({(case["K"], case["M"]) for case in eligible})
                families = sorted({case["field_model"] for case in eligible})
                if test_case_count < len(pairs):
                    raise ValueError("Test sample is too small to cover every K, M pair")
                quotient, remainder = divmod(test_case_count, len(pairs))
                selected = []
                for pair_index, (K, M) in enumerate(pairs):
                    quota = quotient + int(pair_index < remainder)
                    for pick in range(quota):
                        family = families[(pair_index + pick) % len(families)]
                        family_cases = [
                            case
                            for case in eligible
                            if case["field_model"] == family
                            and case["K"] == K
                            and case["M"] == M
                        ]
                        replicate_index = pick // len(families)
                        if len(family_cases) <= replicate_index:
                            raise ValueError(
                                f"{dataset} lacks cases for {(family, K, M)}"
                            )
                        selected.append(family_cases[replicate_index])
            eligible = selected
            if len(eligible) < test_case_count:
                raise RuntimeError(f"Test selection for {dataset} is too small")

        for case in eligible:
            entries.append((dataset, directory / case["file"], case))
    if not entries:
        raise ValueError("No cases match this configuration")
    return entries


def _checkpoint_rows(path):
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                rows[(row["fingerprint"], row["dataset"], row["case"], row["N"], row["method"])] = row
            except (ValueError, KeyError):
                continue  # Ignore an interrupted append.
    return rows


def summarize(rows, methods):
    """Compare only cases completed by every requested method."""
    cases = defaultdict(dict)
    for row in rows:
        cases[(row["dataset"], row["case"], row["N"])][row["method"]] = row
    groups = defaultdict(list)
    gaps = defaultdict(list)
    for group in cases.values():
        if any(m not in group or group[m]["status"] != "ok" for m in methods):
            continue
        exact = group.get("Exhaustive")
        for method, row in group.items():
            key = (row["dimension"], row["budget_level"], method)
            groups[key].append(row)
            if exact:
                gap = row["actual_error"] - exact["actual_error"]
                if gap < -1e-9:
                    raise RuntimeError("A method beat the exhaustive result on a matched case")
                gaps[key].append(max(0.0, gap))
    summary = []
    for (dimension, level, method), group in sorted(groups.items()):
        differences = gaps[(dimension, level, method)]
        summary.append({
            "dimension": dimension, "budget_level": level, "method": method,
            "case_count": len(group), "mean_budget_fraction": fmean(r["budget_fraction"] for r in group),
            "mean_error_probability": fmean(r["actual_error"] for r in group),
            "mean_seconds": fmean(r["seconds"] for r in group),
            "mean_error_gap": fmean(differences) if differences else "",
        })
    return summary


def run_analysis(*, project=PROJECT, study="compact", mode="test", exact_k_max=4,
                 time_limit=30.0, resume=True, beam_width=3, max_iterations=30):
    """Save actual errors, trees, timings, and explicit timeout records."""
    project = Path(project)
    entries = select_cases(project, study, mode, exact_k_max)
    methods = COMPACT_METHODS if study == "compact" else EXTENDED_METHODS
    if not isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite")
    settings = {"study": study, "mode": mode, "exact_k_max": exact_k_max,
                "time_limit": time_limit, "beam_width": beam_width, "max_iterations": max_iterations}
    code = b"".join(p.read_bytes() for p in sorted((PROJECT / "src").glob("*.py")))
    code_hash = sha256(code).hexdigest()
    output = project / "results" / study / mode
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "records.jsonl"
    cached = _checkpoint_rows(checkpoint) if resume else {}
    # A newline prevents a partial final JSON record from swallowing the next append.
    with checkpoint.open("a") as stream:
        stream.write("\n")
    rows, reused = [], 0
    dataset_totals = Counter(dataset for dataset, _, _ in entries)
    dataset_completed = Counter()
    print(f"{study}/{mode}: {len(entries)} cases, methods: {', '.join(methods)}", flush=True)
    if study == "compact":
        print(f"Exact comparison uses K <= {exact_k_max}; limit {time_limit:g}s per method/budget.", flush=True)
    for dataset, path, metadata in entries:
        K, M, phases = load_phase_input(path)
        if K != metadata["K"] or M != metadata["M"]:
            raise ValueError(f"Manifest dimensions disagree with {path}")
        fingerprint = sha256(path.read_bytes() + json.dumps(settings, sort_keys=True).encode()
                             + code_hash.encode() + json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        for level, budget in enumerate(metadata["sensor_budgets"], 1):
            for method in methods:
                key = (fingerprint, dataset, path.stem, budget, method)
                if key in cached:
                    rows.append(cached[key]); reused += 1
                    continue
                seed = crc32(f"{dataset}:{path.stem}:{budget}".encode())
                options = {"time_limit": time_limit}
                if method in ("Heuristic", "Uniform", "Random"):
                    options["random_seed"] = seed
                if method == "Heuristic":
                    options.update(beam_width=beam_width, max_iterations=max_iterations)
                row = {"fingerprint": fingerprint, "dataset": dataset, "case": path.stem,
                       "dimension": metadata["spatial_dimension"], "field_family": metadata["field_model"],
                       "replicate": metadata["replicate"], "K": K, "M": M, "N": budget,
                       "budget_level": level,
                       "budget_fraction": budget / metadata["minimum_sensors_for_zero_error"],
                       "method": method, "status": "ok", "actual_error": None,
                       "allocation": None, "decision_tree": None, "seconds": None}
                begin = perf_counter()
                try:
                    result = SOLVERS[method](K, M, budget, phases, **options)
                    row.update(actual_error=result.actual_error, allocation=list(result.allocation),
                               decision_tree=tree_to_dict(result.decision_tree))
                except SearchTimeout:
                    row["status"] = "timeout"
                row["seconds"] = perf_counter() - begin
                with checkpoint.open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                rows.append(row)
        dataset_completed[dataset] += 1
        completed = dataset_completed[dataset]
        print(
            f"{dataset}: completed case {completed}/{dataset_totals[dataset]} "
            f"({metadata['field_model']}, K={K}, M={M})",
            flush=True,
        )
    summary = summarize(rows, methods)
    flat = []
    for row in rows:
        exported = {
            k: v
            for k, v in row.items()
            if k not in ("fingerprint", "decision_tree", "actual_error")
        }
        exported["allocation"] = (
            json.dumps(row["allocation"]) if row["allocation"] is not None else ""
        )
        exported["error_probability"] = row["actual_error"]
        flat.append(exported)
    write_csv(output / "tables/case_errors.csv", flat)
    write_csv(output / "tables/error_summary.csv", summary)
    timeout_fields = [
        "dataset", "case", "dimension", "field_family", "replicate", "K", "M",
        "N", "budget_level", "budget_fraction", "method", "status", "seconds",
        "time_limit_seconds",
    ]
    timeout_rows = [
        {
            **{field: row[field] for field in timeout_fields if field != "time_limit_seconds"},
            "time_limit_seconds": time_limit,
        }
        for row in rows
        if row["status"] == "timeout"
    ]
    write_csv(output / "tables/timeouts.csv", timeout_rows, fieldnames=timeout_fields)
    counts = Counter((r["dimension"], r["method"], r["status"]) for r in rows)
    write_csv(output / "tables/status.csv", [dict(dimension=d, method=m, status=s, count=n)
              for (d, m, s), n in sorted(counts.items())])
    (output / "run.json").write_text(json.dumps({**settings, "methods": methods,
        "selected_cases": len(entries), "code_hash": code_hash,
        "matched_summary_rows": len(summary), "timeout_count": len(timeout_rows)}, indent=2) + "\n")
    return {"rows": rows, "summary": summary, "output": output}
