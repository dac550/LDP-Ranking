"""
Poisoning Attack Strategies for GRR
=====================================
Three attack strategies targeting the GRR (Generalized Random Response)
protocol for ranking estimation:

  1. Random Attack        — Each fake user randomly reports a target item.
  2. Greedy Attack        — Fake users iteratively promote the most
                            cost-effective non-target item to displace items
                            ranked above target items.
  3. MPOIA               — Like Greedy, but distances account for perturbation
                            noise using a statistical confidence margin.

Threat Model:
    The adversary controls n2 fake users who submit crafted GRR reports.
    The goal is to maximize the total rank gain of target items T by
    injecting fake reports without violating the GRR perturbation format.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_distances(
    target_items: List[int],
    eff_items: List[int],
    freq: Dict[int, int],
) -> Dict[int, int]:
    """
    For each effective attack item, compute the minimum number of additional
    reports needed to make it rank above the closest target item.

    An effective attack item a ∈ A is one for which there exists at least one
    target item t ∈ T with freq[t] > freq[a]. The distance to target t is:
        dist(a, t) = freq[t] - freq[a] + 1

    We store the minimum such distance over all qualifying targets.

    Args:
        target_items: Target item IDs.
        eff_items: Effective attack item IDs.
        freq: Current frequency estimates.

    Returns:
        Dict mapping each effective item → minimum distance to nearest target.
    """
    distances = {}
    for a in eff_items:
        qualifying = [t for t in target_items if freq[t] >= freq[a]]
        if qualifying:
            closest = min(qualifying, key=lambda t: freq[t])
            distances[a] = freq[closest] - freq[a] + 1
    return distances


def _compute_distances_mpoia(
    target_items: List[int],
    eff_items: List[int],
    original_freq: Dict[int, int],
    perturbed_freq: Dict[int, int],
    n: int,
    p: float,
    confidence: float = 0.9,
) -> Dict[int, int]:
    """
    Compute distances with MPOIA statistical confidence margin.

    Instead of using the raw expected perturbed counts, MPOIA accounts for
    the variance of the MI estimator. The distance δ is chosen such that
    the attack item surpasses the target item's count with probability
    ≥ confidence, given the noise in the MI estimator.

    δ = E[D] + z_α * sqrt(Var(D)) + 1
    where D = freq[t] - freq[a] under MI estimation.

    Args:
        target_items: Target item IDs.
        eff_items: Effective attack item IDs.
        original_freq: Original (pre-perturbation) frequency counts.
        perturbed_freq: Expected perturbed frequency estimates.
        n: Total honest users.
        p: GRR keep-probability.
        confidence: One-sided confidence level (default 0.90).

    Returns:
        Dict mapping each effective item → MPOIA distance.
    """
    d = len(perturbed_freq)
    q = (1 - p) / (d - 1)
    z_alpha = norm.ppf(confidence)

    distances = {}
    for a in eff_items:
        qualifying = [t for t in target_items if perturbed_freq[t] >= perturbed_freq[a]]
        if qualifying:
            closest = min(qualifying, key=lambda t: perturbed_freq[t])
            E_D = perturbed_freq[closest] - perturbed_freq[a]
            Var_D = (
                original_freq[closest] * p * (1 - p)
                + (n - original_freq[closest]) * q * (1 - q)
                + original_freq[a] * p * (1 - p)
                + (n - original_freq[a]) * q * (1 - q)
            )
            delta = int(E_D + z_alpha * np.sqrt(Var_D)) + 1
            distances[a] = delta
    return distances

def _compute_distances_vectorized(
        target_items: List[int],
        eff_items: List[int],
        freq: Dict[int, int],
) -> Dict[int, int]:
    """向量化版本的距离计算"""
    # 转换为numpy数组
    freq_array = np.array([freq.get(i, 0) for i in range(max(freq.keys()) + 1)])
    target_freqs = freq_array[target_items]

    distances = {}
    for a in eff_items:
        a_freq = freq[a]
        qualifying_mask = target_freqs >= a_freq
        if np.any(qualifying_mask):
            min_target_freq = np.min(target_freqs[qualifying_mask])
            distances[a] = min_target_freq - a_freq + 1
    return distances

def _effective_items(
    A: List[int],
    T: List[int],
    freq: Dict[int, int],
) -> List[int]:
    """Items in A that are ranked below at least one target item."""
    return [a for a in A if any(freq[t] > freq[a] for t in T)]


# ---------------------------------------------------------------------------
# Public Attack APIs
# ---------------------------------------------------------------------------

def random_attack_grr(n2: int, target_items: List[int]) -> List[int]:
    """
    Random Attack for GRR.

    Each fake user uniformly at random selects and reports one target item.
    This is the simplest baseline attack: it directly inflates target item
    counts in the aggregated histogram, but makes no strategic choices.

    Args:
        n2: Number of fake users to inject.
        target_items: List of target item IDs.

    Returns:
        List of n2 integer values (fake GRR reports), each drawn from target_items.

    Example:
        >>> fake = random_attack_grr(n2=1000, target_items=[2, 5, 8])
    """
    choices = np.random.choice(target_items, n2)
    return [int(x) for x in choices]

from tqdm import tqdm
def greedy_attack_grr(
    n2: int,
    target_items: List[int],
    non_target_items: List[int],
    expected_perturbed_freq: Dict[int, int],
    est_rank_dict: Dict[int, int],
) -> List[int]:
    """
    Greedy Attack for GRR.

    Iteratively identifies "effective attack items" — non-target items ranked
    above at least one target item — and focuses fake users on the one closest
    (fewest reports needed) to a target item. Once an attack item surpasses a
    target item, it no longer blocks that target's promotion.

    This strategy minimizes wasted budget: instead of directly inflating target
    items (which are already tracked), it strategically lowers the apparent
    barrier between non-targets and targets.

    Algorithm:
        while n2 > 0:
            eff = {a ∈ A : ∃ t ∈ T, freq[t] > freq[a]}
            opt = argmin_{a ∈ eff} dist(a, nearest_target)
            inject dist(opt) fake users reporting opt
            update freq[opt]

    Args:
        n2: Number of fake users to inject.
        target_items: List of target item IDs (T).
        non_target_items: List of non-target item IDs (A).
        expected_perturbed_freq: Expected perturbed frequency estimates.
        est_rank_dict: Current estimated rank dictionary.

    Returns:
        List of fake GRR report values (integers in domain).

    Example:
        >>> fake = greedy_attack_grr(n2=500, target_items=[2,5],
        ...     non_target_items=[0,1,3,4,6,...], expected_perturbed_freq=epf,
        ...     est_rank_dict=ranks)
    """
    attacked_freq = expected_perturbed_freq.copy()
    fake_data = []

    # 预转换为 set 加速查找
    target_set = set(target_items)
    non_target_set = set(non_target_items)

    def get_eff_items():
        return [a for a in non_target_set
                if any(attacked_freq.get(t, 0) > attacked_freq.get(a, 0) for t in target_set)]

    eff_items = get_eff_items()

    # 使用 tqdm 进度条
    from tqdm import tqdm
    pbar = tqdm(total=n2, desc="Greedy Attack", unit="fake")

    while n2 > 0:
        distances = _compute_distances(target_set, eff_items, attacked_freq)
        if not distances:
            break

        opt_item = min(distances, key=distances.get)
        dist = distances[opt_item]

        steps = min(dist, n2)
        fake_data.extend([opt_item] * steps)
        attacked_freq[opt_item] = attacked_freq.get(opt_item, 0) + steps
        n2 -= steps
        pbar.update(steps)

        eff_items = get_eff_items()

    pbar.close()

    if n2 > 0:
        fake_data.extend(np.random.choice(list(non_target_set), n2, replace=True).tolist())

    return fake_data


def mpoia_attack_grr(
    n2: int,
    target_items: List[int],
    non_target_items: List[int],
    original_freq: Dict[int, int],
    expected_perturbed_freq: Dict[int, int],
    est_rank_dict: Dict[int, int],
    n: int,
    p: float,
    confidence: float = 0.9,
) -> List[int]:
    """
    Maximum Probability of Overtaking with Influence Awareness (MPOIA) for GRR.

    An improved greedy attack that accounts for the statistical noise introduced
    by the MI estimator. Rather than targeting the raw distance, it computes a
    confidence-adjusted distance δ such that with probability ≥ `confidence`,
    the attack item's estimated count will exceed the target item's count after
    δ additional fake reports.

    This is more robust than the plain greedy attack when n is small and
    perturbation noise is high relative to frequency differences.

    Args:
        n2: Number of fake users.
        target_items: Target item IDs (T).
        non_target_items: Non-target item IDs (A).
        original_freq: True frequency counts (before perturbation).
        expected_perturbed_freq: Expected perturbed frequencies.
        est_rank_dict: Estimated rank dictionary.
        n: Total honest user count.
        p: GRR keep-probability (e^ε / (e^ε + d - 1)).
        confidence: Statistical confidence level for overtaking (default 0.90).

    Returns:
        List of fake GRR report values.

    Example:
        >>> fake = mpoia_attack_grr(n2=500, target_items=[2,5],
        ...     non_target_items=[0,1,3,4,...], original_freq=freq,
        ...     expected_perturbed_freq=epf, est_rank_dict=ranks,
        ...     n=10000, p=0.77)
    """
    attacked_freq = expected_perturbed_freq.copy()
    fake_data = []
    eff_items = _effective_items(non_target_items, target_items, attacked_freq)

    while n2 > 0:
        distances = _compute_distances_mpoia(
            target_items, eff_items, original_freq, attacked_freq, n, p, confidence
        )
        if not distances:
            break

        opt_item = min(distances, key=distances.get)
        dist = distances[opt_item]

        steps = min(dist, n2)
        fake_data.extend([opt_item] * steps)
        attacked_freq[opt_item] += steps
        n2 -= steps

        eff_items = _effective_items(non_target_items, target_items, attacked_freq)

    if n2 > 0:
        fake_data.extend(list(np.random.choice(non_target_items, n2, replace=True)))

    return fake_data
