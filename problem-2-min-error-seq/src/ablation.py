"""Matched allocation and tree ablations using completed extended results."""

from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from statistics import fmean
from zlib import crc32

import matplotlib.pyplot as plt

from analysis import PROJECT, select_cases, write_csv, source_code_hash, compatible_source_digest
from baselines import random_tree
from figures import STYLES, _plot_style
from heuristic import _build_tree_with_beam_search, _compute_pair_errors, _initial_allocation
from input_data import load_phase_input
from strategy import OutcomeModel, tree_to_dict

METHODS = ("Near-Uniform + Beam Search", "Greedy + Beam Search",
           "VND + Beam Search", "VND + Random Tree")
STYLE_KEYS = dict(zip(METHODS, ("Uniform", "Greedy", "Heuristic", "Random")))


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _context(project, mode):
    project = Path(project)
    source = project / "results/extended" / mode
    info = json.loads((source / "run.json").read_text())
    code_hash = source_code_hash(project)
    if info["code_hash"] != code_hash:
        raise ValueError("Run notebook 02 with the current source before notebook 03")
    if info["time_limit"] is not None or info["timeout_count"]:
        raise ValueError("Notebook 03 requires completed extended results with no time limit")
    settings = dict(mode=mode, source_run=info, source_csv_sha256=_digest(source / "tables/case_errors.csv"),
                    ablation_sha256=compatible_source_digest(project, 'ablation', _digest(Path(__file__))))
    output = project / "results/ablation" / mode
    saved = output / "run.json"
    if saved.exists() and json.loads(saved.read_text()) != settings:
        raise ValueError("Saved ablation settings differ; clear this study output before a new run")
    return source, settings, output


def _beam_result(model, allocation, beam_width):
    locations, errors = _compute_pair_errors(model.pairs, model.M, allocation, model.phases)
    outcomes = {pair: model.probabilities(pair, locations[pair], allocation[locations[pair]])
                for pair in model.pairs}
    tree = _build_tree_with_beam_search(model.K, locations, errors, outcomes, beam_width,
                                        check_time=model.check_time)
    return model.result(allocation, tree)


def summarize_ablation(rows):
    """Match all variants and give the two dimensions equal weight."""
    matched = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["case"], int(row["N"]))
        if row["method"] in matched[key]:
            raise ValueError("Duplicate ablation result")
        matched[key][row["method"]] = row
    groups = defaultdict(list)
    for group in matched.values():
        if set(group) != set(METHODS):
            raise ValueError("Incomplete ablation comparison")
        if group[METHODS[2]]["allocation"] != group[METHODS[3]]["allocation"]:
            raise ValueError("Tree ablation must use the same VND allocation")
        for row in group.values():
            groups[(int(row["budget_level"]), row["method"], int(row["dimension"]))].append(row)
    summary = []
    for level in sorted({key[0] for key in groups}):
        for method in METHODS:
            dimensions = [groups[(level, method, d)] for d in (1, 2)]
            if any(not values for values in dimensions):
                raise ValueError("Every budget level requires both dimensions")
            summary.append(dict(budget_level=level, method=method,
                case_count=sum(map(len, dimensions)),
                mean_budget_fraction=fmean(fmean(float(r["budget_fraction"]) for r in values)
                                          for values in dimensions),
                mean_error_probability=fmean(fmean(float(r["actual_error"]) for r in values)
                                            for values in dimensions)))
    return summary


def run_ablation(*, project=PROJECT, mode="full", resume=True):
    """Reuse notebook 02 allocations and checkpoint each instance-budget comparison."""
    source, settings, output = _context(project, mode)
    entries = select_cases(Path(project), "extended", mode, exact_k_max=4)
    with (source / "tables/case_errors.csv").open(newline="") as stream:
        saved = {}
        for row in csv.DictReader(stream):
            if row["method"] in ("Greedy", "Heuristic"):
                key = (row["dataset"], row["case"], int(row["N"]), row["method"])
                if key in saved or row["status"] != "ok":
                    raise ValueError("Notebook 02 must contain unique completed results")
                saved[key] = row
    expected = {(dataset, path.stem, n, method) for dataset, path, meta in entries
                for n in meta["sensor_budgets"] for method in ("Greedy", "Heuristic")}
    if set(saved) != expected:
        raise ValueError("Notebook 02 results do not match the selected extended dataset")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (dataset, path, meta) in enumerate(entries, 1):
        K, M, phases = load_phase_input(path)
        if (K, M) != (meta["K"], meta["M"]):
            raise ValueError("Input dimensions disagree with the manifest")
        input_hash = _digest(path)
        for level, N in enumerate(meta["sensor_budgets"], 1):
            if N <= 0:
                raise ValueError("Ablation budgets must be positive")
            checkpoint = output / "cache" / dataset / path.stem / f"n{N}.json"
            if resume and checkpoint.exists():
                cached = json.loads(checkpoint.read_text())
                if cached["input_sha256"] != input_hash:
                    raise ValueError("Input changed since the cached ablation")
                records.extend(cached["rows"])
                continue
            seed = crc32(f"{dataset}:{path.stem}:{N}".encode())
            model = OutcomeModel(K, M, N, phases)
            near_uniform = _initial_allocation(M, N, "near_uniform", seed)
            previous = {m: saved[(dataset, path.stem, N, m)] for m in ("Greedy", "Heuristic")}
            allocations = {m: tuple(json.loads(row["allocation"])) for m, row in previous.items()}
            for row in previous.values():
                if (int(row["K"]), int(row["M"]), int(row["budget_level"])) != (K, M, level):
                    raise ValueError("Saved result metadata disagrees with the input")
                if abs(float(row["budget_fraction"]) - N / meta["minimum_sensors_for_zero_error"]) > 1e-12:
                    raise ValueError("Saved sensor-budget fraction disagrees with the input")
            results = {
                METHODS[0]: _beam_result(model, near_uniform, settings["source_run"]["beam_width"]),
                METHODS[1]: _beam_result(model, allocations["Greedy"], settings["source_run"]["beam_width"]),
                METHODS[3]: model.result(allocations["Heuristic"],
                    random_tree(model, allocations["Heuristic"], Random(seed))),
            }
            rows, trees = [], {}
            for method in METHODS:
                result = results.get(method)
                allocation = result.allocation if result else allocations["Heuristic"]
                error = result.actual_error if result else float(previous["Heuristic"]["error_probability"])
                rows.append(dict(dataset=dataset, case=path.stem, dimension=meta["spatial_dimension"],
                    field_family=meta["field_model"], K=K, M=M, N=N, budget_level=level,
                    budget_fraction=N / meta["minimum_sensors_for_zero_error"], method=method,
                    allocation=list(allocation), actual_error=error, random_seed=seed))
                if result:
                    trees[method] = tree_to_dict(result.decision_tree)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps(dict(input_sha256=input_hash, rows=rows, trees=trees), allow_nan=False))
            temporary.replace(checkpoint)
            records.extend(rows)
        print(f"Ablation: completed case {index}/{len(entries)} ({dataset}, {meta['field_model']}, K={K}, M={M})", flush=True)
    summary = summarize_ablation(records)
    write_csv(output / "tables/case_errors.csv", [dict(r, allocation=json.dumps(r["allocation"])) for r in records])
    write_csv(output / "tables/error_summary.csv", summary)
    (output / "run.json").write_text(json.dumps(settings, indent=2) + "\n")
    return dict(summary=summary, output=output)


def load_ablation(*, project=PROJECT, mode="full"):
    _, _, output = _context(project, mode)
    path = output / "tables/error_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run notebook 03 in {mode} mode first: {path}")
    with path.open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    return dict(summary=summary, output=output)


def plot_ablation(summary):
    _plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    panels = ((METHODS[:3], "(a) Sensor allocation"),
              ((METHODS[3], METHODS[2]), "(b) Test ordering"))
    for axis, (methods, title) in zip(axes, panels):
        for method in methods:
            rows = sorted((r for r in summary if r["method"] == method), key=lambda r: int(r["budget_level"]))
            color, marker, line = STYLES[STYLE_KEYS[method]]
            axis.plot([float(r["mean_budget_fraction"]) for r in rows],
                      [float(r["mean_error_probability"]) for r in rows],
                      label=method, color=color, marker=marker, linestyle=line, linewidth=1.8)
        axis.set_title(title, fontsize=12)
        axis.set_xlabel(r"Mean budget fraction, $N/N_{\min}^{\mathrm{seq}}$")
        axis.grid(alpha=0.22)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=10)
    axes[0].set_ylabel("Mean error probability")
    axes[0].set_ylim(bottom=0)
    figure.subplots_adjust(top=0.88, bottom=0.34, wspace=0.08)
    return figure
