"""
LDP Poison Toolkit
==================
A research toolkit for evaluating poisoning attacks on Local Differential
Privacy (LDP) protocols for ranking estimation.

Supported Protocols:
    - GRR  : Generalized Random Response
    - OUE  : Optimized Unary Encoding
    - OLH  : Optimized Local Hashing

Supported Attacks:
    - Random Attack  : Fake users report target items directly.
    - ROA            : Random Output Attack (noise baseline).
    - Greedy Attack  : Iterative cost-minimizing attack.
    - MPOIA          : Greedy with statistical confidence margins (GRR only).

Quick Start:
    >>> from ldp_poison_toolkit.experiments import AttackRunner
    >>> runner = AttackRunner(n=10000, d=50, epsilon=2.0, n2=1000, r=3, x=10)
    >>> result = runner.run(protocol='grr', attack='greedy')
    >>> print(result.summary())

Based on protocols from the Multi-Freq-LDPy package.
"""

from .experiments.runner import AttackRunner, AttackResult
from .experiments.benchmark import benchmark_attacks
from .core import GRR_Client, GRR_Aggregator_MI, UE_Client, UE_Aggregator_MI, LH_Client, LH_Aggregator_MI
from .utils import (
    generate_dataset,
    generate_freq_dict,
    generate_target_items,
    get_gain,
    get_rank_gain,
    grr_estimate,
    oue_estimate,
    olh_estimate,
    expected_perturbed_frequencies,
)

__version__ = "1.0.0"
__author__ = "LDP Poison Toolkit Contributors"
__license__ = "MIT"

__all__ = [
    # High-level API
    "AttackRunner",
    "AttackResult",
    "benchmark_attacks",
    # Core protocols
    "GRR_Client", "GRR_Aggregator_MI",
    "UE_Client", "UE_Aggregator_MI",
    "LH_Client", "LH_Aggregator_MI",
    # Utilities
    "generate_dataset",
    "generate_freq_dict",
    "generate_target_items",
    "get_gain",
    "get_rank_gain",
    "grr_estimate",
    "oue_estimate",
    "olh_estimate",
    "expected_perturbed_frequencies",
]
