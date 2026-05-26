"""
Optimized Local Hashing (OLH) Protocol
========================================
Local Hashing protocol [Wang et al., 2017], which hashes the input into a
smaller domain of size g = e^ε + 1 before applying GRR.

References:
    [1] Wang et al. (2017) "Locally differentially private protocols for frequency
        estimation" (USENIX Security).
    [2] Bassily and Smith "Local, private, efficient protocols for succinct
        histograms" (STOC).
"""

import numpy as np
from sys import maxsize
import xxhash

from .grr import GRR_Client


def _matrix_inversion(count_report: np.ndarray, n: int, p: float, q: float) -> np.ndarray:
    """Matrix Inversion (MI) frequency estimator."""
    est_freq = np.array((count_report - n * q) / (p - q)).clip(0)
    return np.round(est_freq)


def LH_Client(input_data: int, d: int, epsilon: float, optimal: bool = True) -> tuple:
    """
    OLH client-side perturbation for a single user value.

    Each user:
      1. Picks a random hash function h: [d] → [g] (identified by a seed).
      2. Computes y = h(input_data) ∈ [0, g-1].
      3. Applies GRR on y within the reduced domain [g].

    With optimal=True (OLH): g = round(e^ε) + 1.
    With optimal=False (BLH): g = 2.

    Args:
        input_data: The user's true value in [0, d-1].
        d: Domain size.
        epsilon: Privacy budget (ε > 0).
        optimal: If True, use Optimized LH (OLH).

    Returns:
        Tuple of (sanitized_value, rnd_seed) where sanitized_value ∈ [0, g-1].

    Raises:
        ValueError: If input_data is out of range, d < 2, or epsilon <= 0.

    Example:
        >>> report = LH_Client(input_data=3, d=100, epsilon=2.0)
    """
    if input_data < 0 or input_data >= d:
        raise ValueError(f"input_data must be in [0, d-1], got {input_data}.")
    if not isinstance(d, int) or d < 2:
        raise ValueError("d must be an integer >= 2.")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")

    g = int(round(np.exp(epsilon))) + 1 if optimal else 2

    rnd_seed = np.random.randint(0, maxsize, dtype=np.int64)
    hashed_input = xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % g
    sanitized_value = GRR_Client(hashed_input, g, epsilon)

    return (sanitized_value, rnd_seed)


def LH_Aggregator_MI(reports: list, d: int, epsilon: float, optimal: bool = True) -> np.ndarray:
    """
    OLH server-side aggregator using Matrix Inversion (MI).

    For each report (y, seed), counts for value v are incremented whenever
    hash_seed(v) == y. MI is then applied using OLH's p and q parameters.

    Args:
        reports: List of (sanitized_value, rnd_seed) tuples from LH_Client.
        d: Domain size.
        epsilon: Privacy budget.
        optimal: If True, use OLH parameters.

    Returns:
        Array of estimated frequencies (rounded, non-negative).

    Raises:
        ValueError: If reports is empty, d < 2, or epsilon <= 0.

    Example:
        >>> estimates = LH_Aggregator_MI(reports=tuples, d=100, epsilon=2.0)
    """
    if len(reports) == 0:
        raise ValueError("reports list is empty.")
    if not isinstance(d, int) or d < 2:
        raise ValueError("d must be an integer >= 2.")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")

    n = len(reports)
    g = int(round(np.exp(epsilon))) + 1 if optimal else 2

    count_report = np.zeros(d)
    for sanitized_value, rnd_seed in reports:
        for v in range(d):
            if sanitized_value == xxhash.xxh32(str(v), seed=rnd_seed).intdigest() % g:
                count_report[v] += 1

    p = np.exp(epsilon) / (np.exp(epsilon) + g - 1)
    q = 1.0 / g

    return _matrix_inversion(count_report, n, p, q)
