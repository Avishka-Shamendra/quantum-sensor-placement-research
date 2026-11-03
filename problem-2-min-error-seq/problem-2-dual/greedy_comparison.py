"""Compatibility entry points for the common resumable dual-study runner."""
from dual_search import TARGETS


def run_greedy_study(mode='test', **options):
    from budget_limited import run_comparison
    return run_comparison(mode, max_sensor_bound=None, methods=('greedy',), **options)


def run_comparison(mode='test', **options):
    from budget_limited import run_comparison as run
    return run(mode, max_sensor_bound=None, **options)
