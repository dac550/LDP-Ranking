"""
Optimized Unary Encoding (OUE) Protocol
========================================
A variant of RAPPOR [Erlingsson et al., 2014] optimized for frequency estimation
[Wang et al., 2017].

References:
    [1] Erlingsson, Pihur, and Korolova (2014) "RAPPOR: Randomized aggregatable
        privacy-preserving ordinal response" (ACM CCS).
    [2] Wang et al. (2017) "Locally differentially private protocols for frequency
        estimation" (USENIX Security).
"""

import numpy as np


def _matrix_inversion(count_report: np.ndarray, n: int, p: float, q: float) -> np.ndarray:
    """Matrix Inversion (MI) frequency estimator."""
    est_freq = np.array((count_report - n * q) / (p - q)).clip(0)
    return np.round(est_freq)

from scipy.sparse import lil_matrix, csr_matrix

def UE_Client(input_data: int, d: int, epsilon: float, optimal: bool = True,return_sparse:bool=False) -> np.ndarray:
    """
    OUE client-side perturbation for a single user value.

    Encodes the user's value as a one-hot binary vector of length d, then
    perturbs each bit independently:
      - Bits set to 1: flipped to 0 with probability (1 - p).
      - Bits set to 0: flipped to 1 with probability q.

    With optimal=True (OUE): p = 1/2, q = 1/(e^ε + 1).
    With optimal=False (RAPPOR): p = e^(ε/2)/(e^(ε/2)+1), q = 1 - p.

    Args:
        input_data: The user's true value in [0, d-1].
        d: Domain size.
        epsilon: Privacy budget (ε > 0).
        optimal: If True, use Optimized UE parameters (recommended).

    Returns:
        A binary vector of length d representing the privatized report.

    Raises:
        ValueError: If input_data is out of range, d < 2, or epsilon <= 0.

    Example:
        >>> vec = UE_Client(input_data=3, d=10, epsilon=2.0)
    """
    if input_data is not None:
        if input_data < 0 or input_data >= d:
            raise ValueError(f"input_data must be in [0, d-1], got {input_data}.")
    if not isinstance(d, int) or d < 2:
        raise ValueError("d must be an integer >= 2.")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")

    if optimal:
        p = 0.5
        q = 1.0 / (np.exp(epsilon) + 1)
    else:
        p = np.exp(epsilon / 2) / (np.exp(epsilon / 2) + 1)
        q = 1.0 - p

    # One-hot encode
    input_ue_data = np.zeros(d)
    if input_data is not None:
        input_ue_data[input_data] = 1

    sanitized_vec = np.zeros(d)
    for ind in range(d):
        rnd = np.random.random()
        threshold = p if input_ue_data[ind] == 1 else q
        if rnd <= threshold:
            sanitized_vec[ind] = 1
    if return_sparse:
        return csr_matrix(sanitized_vec)

    return sanitized_vec


def UE_Aggregator_MI(reports: list, d: int, epsilon: float, optimal: bool = True) -> np.ndarray:
    """
    OUE server-side aggregator using Matrix Inversion (MI).

    Sums all binary vectors from clients and applies the MI estimator to
    obtain unbiased frequency estimates.

    Args:
        reports: List of binary vectors (each of length d) from UE_Client.
        d: Domain size.
        epsilon: Privacy budget.
        optimal: If True, use OUE parameters.

    Returns:
        Array of estimated frequencies (rounded, non-negative).

    Raises:
        ValueError: If reports is empty or epsilon <= 0.

    Example:
        >>> estimates = UE_Aggregator_MI(reports=vecs, d=10, epsilon=2.0)
    """
    if len(reports) == 0:
        raise ValueError("reports list is empty.")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")

    n = len(reports)

    if optimal:
        p = 0.5
        q = 1.0 / (np.exp(epsilon) + 1)
    else:
        p = np.exp(epsilon / 2) / (np.exp(epsilon / 2) + 1)
        q = 1.0 - p

    return _matrix_inversion(sum(reports), n, p, q)
