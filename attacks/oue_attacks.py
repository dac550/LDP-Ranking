"""
Poisoning Attack Strategies for OUE
=====================================
Three attack strategies targeting the OUE (Optimized Unary Encoding) protocol.

Unlike GRR where a fake report is a single integer, in OUE a report is a
binary vector of length d. The adversary crafts binary vectors to inflate
the count of target items.

  1. Random Attack  — Fake users report a target item honestly (1-hot vector
                      encoded by UE_Client).
  2. ROA            — Random Output Attack: fake users report binary vectors
                      with E_1 bits set to 1, chosen from non-target positions.
                      This is used as a baseline that avoids helping targets
                      directly but tests indirect effects.
  3. Greedy Attack  — Selects the E_1 effective attack items with smallest
                      distances and concentrates fake vectors on them.

Parameter:
    E_1 = floor(0.5 + (d - 1) / (e^ε + 1))  — expected number of 1-bits in
    a perturbed non-active position under OUE.
"""

import numpy as np
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_distances(
    target_items: List[int],
    eff_items: List[int],
    freq: Dict[int, int],
) -> Dict[int, int]:
    distances = {}
    for a in eff_items:
        qualifying = [t for t in target_items if freq[t] >= freq[a]]
        if qualifying:
            closest = min(qualifying, key=lambda t: freq[t])
            distances[a] = freq[closest] - freq[a] + 1
    return distances


def _effective_items(A, T, freq):
    return [a for a in A if any(freq[t] > freq[a] for t in T)]


def _e1(d: int, epsilon: float) -> int:
    """Expected number of 1-bits for non-active positions under OUE."""
    return int(0.5 + (d - 1) / (np.exp(epsilon) + 1))


# ---------------------------------------------------------------------------
# Public Attack APIs
# ---------------------------------------------------------------------------

def random_attack_oue(
    n2: int,
    target_items: List[int],
    d: int,
) -> List[np.ndarray]:
    """
    Random Attack for OUE.

    Each fake user uniformly selects one target item and submits a one-hot
    binary vector with a 1 at that item's position. This directly inflates
    the count for each target item equally.

    Note: Unlike GRR random attack, this produces binary vectors (the OUE
    report format), not integer values.

    Args:
        n2: Number of fake users.
        target_items: List of target item IDs.
        d: Domain size.

    Returns:
        List of n2 binary vectors (numpy arrays of length d).

    Example:
        >>> fake_vecs = random_attack_oue(n2=500, target_items=[2,5], d=50)
    """
    fake_data = []
    for _ in range(n2):
        vec = np.zeros(d)
        chosen = np.random.choice(target_items)
        vec[chosen] = 1
        fake_data.append(vec)
    return fake_data


def roa_attack_oue(
    n2: int,
    non_target_items: List[int],
    d: int,
    epsilon: float,
) -> List[np.ndarray]:
    """
    Random Output Attack (ROA) for OUE.

    Each fake user reports E_1 randomly selected non-target positions as 1.
    This attack acts as a noise baseline: it inflates non-target counts
    equally, potentially lowering the relative standing of target items and
    serving as a comparative baseline for targeted attacks.

    Args:
        n2: Number of fake users.
        non_target_items: List of non-target item IDs.
        d: Domain size.
        epsilon: Privacy budget (used to compute E_1).

    Returns:
        List of n2 binary vectors.

    Example:
        >>> fake_vecs = roa_attack_oue(n2=500, non_target_items=[0,1,3,...],
        ...     d=50, epsilon=2.0)
    """
    E1 = _e1(d, epsilon)
    E1 = min(E1, len(non_target_items))

    fake_data = []
    for _ in range(n2):
        vec = np.zeros(d, dtype=int)
        chosen = np.random.choice(non_target_items, E1, replace=False)
        vec[chosen] = 1
        fake_data.append(vec)
    return fake_data


def greedy_attack_oue(
    n2: int,
    target_items: List[int],
    non_target_items: List[int],
    expected_perturbed_freq: Dict[int, int],
    est_rank_dict: Dict[int, int],
    d: int,
    epsilon: float,
) -> List[np.ndarray]:
    """
    Greedy Attack for OUE.

    At each step, identifies the E_1 effective attack items with the smallest
    total distance to target items and constructs a binary vector with 1s at
    those positions. This concentrates the fake users' impact on items most
    likely to overtake target items.

    Note: Because OUE vectors support multiple positions at once (E_1 positions),
    the attack can simultaneously promote multiple effective items per fake user.
    This is a key advantage over GRR where each report supports only one value.

    Algorithm:
        while n2 > 0:
            eff = {a ∈ A : ∃ t ∈ T, freq[t] > freq[a]}
            select E_1 items from eff with smallest distances
            build perturbation vector with 1s at selected positions
            inject delta_opt copies of this vector

    Args:
        n2: Number of fake users.
        target_items: Target item IDs (T).
        non_target_items: Non-target item IDs (A).
        expected_perturbed_freq: Expected perturbed frequency estimates.
        est_rank_dict: Estimated rank dictionary.
        d: Domain size.
        epsilon: Privacy budget.

    Returns:
        List of binary vectors (numpy arrays of length d).

    Example:
        >>> fake_vecs = greedy_attack_oue(n2=500, target_items=[2,5],
        ...     non_target_items=list(range(50)), expected_perturbed_freq=epf,
        ...     est_rank_dict=ranks, d=50, epsilon=2.0)
    """
    attacked_freq = expected_perturbed_freq.copy()
    fake_data = []
    E1 = _e1(d, epsilon)
    eff_items = _effective_items(non_target_items, target_items, attacked_freq)

    while n2 > 0:
        distances = _compute_distances(target_items, eff_items, attacked_freq)
        if not distances:
            break

        opt_item = min(distances, key=distances.get)
        delta_opt = distances[opt_item]

        # Select E_1 items with smallest distances
        sorted_eff = sorted(eff_items, key=lambda x: distances.get(x, float("inf")))
        selected = sorted_eff[:E1]

        perturbation = np.zeros(d)
        perturbation[selected] = 1

        steps = min(delta_opt, n2)
        fake_data.extend([perturbation.copy()] * steps)
        for item in selected:
            attacked_freq[item] += steps
        n2 -= steps

        eff_items = _effective_items(non_target_items, target_items, attacked_freq)

    if n2 > 0:
        for _ in range(n2):
            vec = np.zeros(d)
            chosen = np.random.choice(non_target_items, min(E1, len(non_target_items)), replace=False)
            vec[chosen] = 1
            fake_data.append(vec)

    return fake_data


def greedy_attack_oue_vectorized(n2, target_items, non_target_items, expected_perturbed_freq,
                                 est_rank_dict, d, epsilon):
    """
    向量化版本的贪婪攻击（性能优化）
    """
    attacked_freq = np.array([expected_perturbed_freq.get(i, 0) for i in range(d)])
    fake_data = []
    E1 = _e1(d, epsilon)

    # 使用 NumPy 数组操作
    target_mask = np.isin(np.arange(d), target_items)
    non_target_mask = ~target_mask

    while n2 > 0:
        # 计算所有非目标项的距离（向量化）
        distances = np.where(
            non_target_mask,
            np.maximum(0, attacked_freq[target_items].max() - attacked_freq + 1),
            np.inf
        )

        if np.all(distances == np.inf):
            break

        # 选择距离最小的 E1 个项目
        sorted_indices = np.argsort(distances)
        selected = sorted_indices[:E1]

        # 构建扰动向量
        perturbation = np.zeros(d)
        perturbation[selected] = 1

        steps = min(int(distances[selected[0]]), n2)
        fake_data.extend([perturbation.copy()] * steps)
        attacked_freq[selected] += steps
        n2 -= steps

    return fake_data


def _compute_distances_vectorized_oue(
        target_items: List[int],
        eff_items: List[int],
        freq: Dict[int, int],
) -> np.ndarray:
    """向量化计算所有有效攻击项的距离"""
    max_idx = max(max(target_items), max(eff_items)) if eff_items else max(target_items)
    freq_array = np.array([freq.get(i, 0) for i in range(max_idx + 1)])

    eff_items_array = np.array(eff_items)
    eff_freqs = freq_array[eff_items_array]

    target_freqs = freq_array[target_items]
    target_max = np.max(target_freqs)

    # 广播计算
    distances = target_max - eff_freqs + 1
    distances = np.maximum(0, distances)

    return dict(zip(eff_items, distances))