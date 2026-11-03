"""Experiment settings and atomic cache serialization."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import os
import tempfile

FAMILIES = ('gaussian', 'sinusoidal', 'quadratic')
FRACTIONS = tuple(i / 10 for i in range(1, 10))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False,
                                         encoding='utf-8') as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Config:
    mode: str = 'test'
    seed: int = 20261018
    beam_width: int = 3
    sequential_max_iterations: int | None = 30
    sequential_algorithm: str = 'vnd_beam'
    joint_algorithm: str = 'greedy'
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 20261019

    def __post_init__(self):
        if self.sequential_algorithm not in ('vnd_beam', 'greedy'):
            raise ValueError('sequential_algorithm must be vnd_beam or greedy')
        if self.joint_algorithm not in ('greedy', 'vnd'):
            raise ValueError('joint_algorithm must be greedy or vnd')
        if self.mode not in ('test', 'compact', 'full'):
            raise ValueError("mode must be 'compact' or 'full' ('test' is an alias)")
        if self.beam_width < 1 or self.bootstrap_samples < 2:
            raise ValueError('beam_width must be positive; bootstrap_samples >= 2')
        if self.sequential_max_iterations is not None and self.sequential_max_iterations < 0:
            raise ValueError('sequential_max_iterations must be nonnegative or None')

    @property
    def sizes(self):
        return tuple(range(2, 11)) if self.mode == 'full' else (2, 3)

    @property
    def replicates(self):
        return 15 if self.mode == 'full' else 2

    def to_dict(self):
        return asdict(self)


def budget_schedule(bound):
    """Round halves upward, omit zero, and retain fraction aliases per budget."""
    maximum = min(bound - 1, 50)
    entries = {}
    for tenth in range(1, 10):
        # Integer arithmetic avoids floating-point rounding at half-integers.
        count = (tenth * maximum + 5) // 10
        if count > 0 and count <= maximum:
            entries.setdefault(count, []).append(tenth / 10)
    return maximum, [{'N': n, 'target_fractions': fs} for n, fs in entries.items()]
