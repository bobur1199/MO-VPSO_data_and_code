"""Paths and environment registry shared by the experiment scripts."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CORE = os.path.join(ROOT, 'code', 'core')
DATA = os.path.join(ROOT, 'data')
FRONTS = os.path.join(DATA, 'fronts')
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from vpso_core import ENVIRONMENTS                    # noqa: E402
from hard_envs import HARD_ENVIRONMENTS               # noqa: E402

#: All ten environments keyed as in the manuscript (H6-H10, O1-O5).
ALL_ENVS = {**{f'H{k}': v for k, v in HARD_ENVIRONMENTS.items()},
            **{f'O{k}': v for k, v in ENVIRONMENTS.items()}}
ENV_ORDER = ['H6', 'H7', 'H8', 'H9', 'H10', 'O1', 'O2', 'O3', 'O4', 'O5']


def data_path(name):
    os.makedirs(DATA, exist_ok=True)
    return os.path.join(DATA, name)


def front_path(name):
    os.makedirs(FRONTS, exist_ok=True)
    return os.path.join(FRONTS, name)


def parse_envs(argv, default=ENV_ORDER):
    """Optional first command-line argument: comma-separated environment keys."""
    return argv[1].split(',') if len(argv) > 1 and argv[1] else list(default)
