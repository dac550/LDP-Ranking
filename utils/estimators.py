"""
Protocol Estimation Wrappers
=============================
Convenience wrappers that combine perturbation + aggregation for each LDP
protocol, returning (perturbed_dataset, freq_dict, rank_dict).
"""

import numpy as np
from typing import Dict, List, Tuple

from ..core.grr import GRR_Client, GRR_Aggregator_MI
from ..core.oue import UE_Client, UE_Aggregator_MI
from ..core.olh import LH_Client, LH_Aggregator_MI
from .metrics import compute_rank_dict


def _to_freq_and_rank(est_freq: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int]]:
    freq_dict = {v: int(c) for v, c in enumerate(est_freq)}
    rank_dict = compute_rank_dict(freq_dict)
    return freq_dict, rank_dict


def grr_estimate(
    dataset,
    d: int,
    epsilon: float,
    perturb: bool = True,
) -> Tuple[list, Dict[int, int], Dict[int, int]]:
    """
    Run GRR perturbation (optional) and aggregation on a dataset.

    Args:
        dataset: Iterable of integer values in [0, d-1].
        d: Domain size.
        epsilon: Privacy budget.
        perturb: If True, apply GRR perturbation to each user value.
                 If False, treat dataset as already-perturbed reports.

    Returns:
        Tuple of (perturbed_dataset, freq_dict, rank_dict).

    Example:
        >>> per_data, freq, ranks = grr_estimate(dataset, d=50, epsilon=2.0)
    """
    if perturb:
        per_dataset = [GRR_Client(v, d, epsilon) for v in dataset]
    else:
        per_dataset = list(dataset)

    est_freq = GRR_Aggregator_MI(per_dataset, d, epsilon)
    freq_dict, rank_dict = _to_freq_and_rank(est_freq)
    return per_dataset, freq_dict, rank_dict


def oue_estimate(
    dataset,
    d: int,
    epsilon: float,
    perturb: bool = True,
) -> Tuple[list, Dict[int, int], Dict[int, int]]:
    """
    Run OUE perturbation (optional) and aggregation on a dataset.

    Args:
        dataset: Iterable of integer values in [0, d-1] (if perturb=True),
                 or list of binary vectors (if perturb=False).
        d: Domain size.
        epsilon: Privacy budget.
        perturb: If True, apply OUE perturbation to each user value.

    Returns:
        Tuple of (perturbed_dataset, freq_dict, rank_dict).

    Example:
        >>> per_data, freq, ranks = oue_estimate(dataset, d=50, epsilon=2.0)
    """
    if perturb:
        per_dataset = [UE_Client(v, d, epsilon) for v in dataset]
    else:
        per_dataset = list(dataset)

    est_freq = UE_Aggregator_MI(per_dataset, d, epsilon)
    freq_dict, rank_dict = _to_freq_and_rank(est_freq)
    return per_dataset, freq_dict, rank_dict


def olh_estimate(
    dataset,
    d: int,
    epsilon: float,
    perturb: bool = True,
) -> Tuple[list, Dict[int, int], Dict[int, int]]:
    """
    Run OLH perturbation (optional) and aggregation on a dataset.

    Args:
        dataset: Iterable of integer values in [0, d-1] (if perturb=True),
                 or list of (sanitized_value, rnd_seed) tuples (if perturb=False).
        d: Domain size.
        epsilon: Privacy budget.
        perturb: If True, apply OLH perturbation to each user value.

    Returns:
        Tuple of (perturbed_dataset, freq_dict, rank_dict).

    Example:
        >>> per_data, freq, ranks = olh_estimate(dataset, d=50, epsilon=2.0)
    """
    if perturb:
        per_dataset = [LH_Client(v, d, epsilon) for v in dataset]
    else:
        per_dataset = list(dataset)

    est_freq = LH_Aggregator_MI(per_dataset, d, epsilon)
    freq_dict, rank_dict = _to_freq_and_rank(est_freq)
    return per_dataset, freq_dict, rank_dict


def expected_perturbed_frequencies(
    original_freq: Dict[int, int],
    n: int,
    p: float,
    q: float,
) -> Dict[int, int]:
    """
    Compute the expected perturbed frequency for each item under an LDP mechanism.

    Under the GRR/OUE/OLH channel:
        E[count_v'] = count_v * p + (n - count_v) * q

    This is used by greedy attack strategies to estimate the effect of
    adding fake users without running the full protocol.

    Args:
        original_freq: True frequency dictionary (item → count).
        n: Total number of honest users.
        p: Probability of a true '1' bit remaining '1' (or true value reported).
        q: Probability of a false '0' bit flipping to '1' (or wrong value reported).

    Returns:
        Dictionary of expected perturbed frequencies (item → expected count).

    Example:
        >>> exp_freq = expected_perturbed_frequencies(freq, n=10000, p=0.5, q=0.01)
    """
    return {
        item: int(np.round(count * p + (n - count) * q))
        for item, count in original_freq.items()
    }
