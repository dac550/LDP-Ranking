"""
Generalized Random Response (GRR) Protocol
===========================================
Also known as Direct Encoding [Wang et al., 2017] or k-RR [Kairouz et al., 2016].

References:
    [1] Wang et al. (2017) "Locally differentially private protocols for frequency
        estimation" (USENIX Security).
    [2] Kairouz, Bonawitz, and Ramage (2016) "Discrete distribution estimation under
        local privacy" (ICML).
"""

import numpy as np


def _matrix_inversion(count_report: np.ndarray, n: int, p: float, q: float) -> np.ndarray:
    """
    Matrix Inversion (MI) frequency estimator.

    Computes unbiased frequency estimates from perturbed reports using the
    inverse of the LDP channel matrix.

    Args:
        count_report: Array recording how many times each value was reported.
        n: Total number of reports.
        p: Probability of reporting the true value.
        q: Probability of reporting any other value.

    Returns:
        Rounded non-negative frequency estimates.
    """
    est_freq = np.array((count_report - n * q) / (p - q)).clip(0)
    return np.round(est_freq)

from functools import lru_cache
@lru_cache(maxsize=128)
def _get_grr_params(d: int, epsilon: float) -> float:
    """缓存GRR参数计算"""
    return np.exp(epsilon) / (np.exp(epsilon) + d - 1)

def GRR_Client(input_data: int, d: int, epsilon: float) -> int:
    """
    GRR client-side perturbation for a single user value.

    Each user independently perturbs their true value using the GRR mechanism:
    with probability p = e^ε / (e^ε + d - 1) the true value is reported;
    otherwise a uniformly random value from the remaining domain is chosen.

    Args:
        input_data: The user's true value, must be in [0, d-1].
        d: Domain size (number of distinct values).
        epsilon: Privacy budget (ε > 0).

    Returns:
        A privatized value in [0, d-1].

    Raises:
        ValueError: If input_data is out of range, d < 2, or epsilon <= 0.

    Example:
        >>> privatized = GRR_Client(input_data=3, d=10, epsilon=1.0)
    """
    # 新增类型检查
    if not isinstance(input_data, (int, np.integer)):
        raise TypeError(f"input_data must be integer, got {type(input_data)}")
    if not isinstance(d, (int, np.integer)):
        raise TypeError(f"d must be integer, got {type(d)}")
    if not isinstance(epsilon, (float, int, np.floating)):
        raise TypeError(f"epsilon must be numeric, got {type(epsilon)}")

    if input_data < 0 or input_data >= d:
        raise ValueError(f"input_data must be in [0, d-1], got {input_data}.")
    if not isinstance(d, int) or d < 2:
        raise ValueError("d must be an integer >= 2.")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")

    p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    domain = np.arange(d)

    if np.random.binomial(1, p) == 1:
        return input_data
    else:
        return int(np.random.choice(domain[domain != input_data]))


def GRR_Aggregator_MI(reports: list, d: int, epsilon: float) -> np.ndarray:
    """
    GRR server-side aggregator using Matrix Inversion (MI).

    Collects all perturbed reports and produces an unbiased frequency estimate
    for each value in the domain.

    Args:
        reports: List of privatized values (integers in [0, d-1]).
        d: Domain size.
        epsilon: Privacy budget.

    Returns:
        Array of estimated frequencies (rounded, non-negative) for each value.

    Raises:
        ValueError: If reports is empty, d < 2, or epsilon <= 0.
        IndexError: If a report value is out of [0, d-1].

    Example:
        >>> estimates = GRR_Aggregator_MI(reports=[2, 5, 3, 3, 7], d=10, epsilon=1.0)
    """
    if len(reports) == 0:
        raise ValueError("reports list is empty.")
    if not isinstance(d, int) or d < 2:
        raise ValueError("d must be an integer >= 2.")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")

    n = len(reports)
    p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    q = (1 - p) / (d - 1)

    count_report = np.zeros(d)
    for rep in reports:
        if rep < 0 or rep >= d:
            raise IndexError(f"Report value {rep} is out of bounds for domain size {d}.")
        count_report[rep] += 1

    return _matrix_inversion(count_report, n, p, q)


# 优化：添加向量化的批量客户端处理函数
def GRR_Client_Batch(input_data: np.ndarray, d: int, epsilon: float) -> np.ndarray:
    """
    批量扰动多个用户数据（向量化实现）

    Args:
        input_data: 用户数据数组
        d: 域大小
        epsilon: 隐私预算

    Returns:
        扰动后的值数组
    """
    n = len(input_data)
    p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    domain = np.arange(d)

    keep_mask = np.random.binomial(1, p, n).astype(bool)
    results = input_data.copy()

    # 需要扰动的索引
    perturb_indices = np.where(~keep_mask)[0]
    for idx in perturb_indices:
        choices = domain[domain != input_data[idx]]
        results[idx] = np.random.choice(choices)

    return results