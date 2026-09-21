"""Shared test/full selection and short output paths for analyses 01-03."""

from pathlib import Path


def validate_run_mode(run_mode):
    if run_mode not in ("test", "full"):
        raise ValueError("run_mode must be 'test' or 'full'")


def select_cases(entries, run_mode):
    """Test: all K/families, M=3 or 6, replicate 1 (132 benchmark cases)."""
    validate_run_mode(run_mode)
    if run_mode == "full":
        return entries
    selected = [(path, metadata) for path, metadata in entries
                if int(metadata["M"]) in (3, 6) and int(metadata["replicate"]) == 1]
    if not selected:
        raise ValueError("No test cases found: requires M=3 or 6 and replicate=1")
    return selected


def output_directory(project_directory, analysis, run_mode):
    validate_run_mode(run_mode)
    return Path(project_directory) / "results" / analysis / run_mode
