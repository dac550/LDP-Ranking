"""
Attack Evaluation Metrics
==========================
Functions to measure the effectiveness of poisoning attacks on LDP-based
ranking estimation.
"""

from typing import Dict, List


def get_gain(
    target_items: List[int],
    before_dict: Dict[int, int],
    after_dict: Dict[int, int],
) -> int:
    """
    Compute total frequency gain for target items after an attack.

    Measures how much the aggregate estimated frequency of all target items
    increased as a result of injecting fake users.

    Args:
        target_items: List of target item IDs.
        before_dict: Frequency dictionary before the attack.
        after_dict: Frequency dictionary after the attack.

    Returns:
        Total frequency gain summed over all target items.
        Positive values indicate the attack successfully boosted frequencies.

    Example:
        >>> gain = get_gain(targets, original_freq, attacked_freq)
    """
    gains = [
        after_dict.get(item, 0) - before_dict.get(item, 0)
        for item in target_items
    ]
    return sum(gains)


def get_rank_gain(
    target_items: List[int],
    before_rank_dict: Dict[int, int],
    after_rank_dict: Dict[int, int],
) -> int:
    """
    Compute total rank improvement for target items after an attack.

    Rank gain is defined as (rank_before - rank_after) summed over all target
    items. A positive value means target items moved up in the ranking.

    Args:
        target_items: List of target item IDs.
        before_rank_dict: Rank dictionary before the attack (lower rank = more popular).
        after_rank_dict: Rank dictionary after the attack.

    Returns:
        Total rank improvement summed over all target items.

    Example:
        >>> rank_gain = get_rank_gain(targets, before_ranks, after_ranks)
    """
    gains = [
        before_rank_dict.get(item, 0) - after_rank_dict.get(item, 0)
        for item in target_items
    ]
    return sum(gains)


def compute_rank_dict(freq_dict: Dict[int, int]) -> Dict[int, int]:
    """
    Convert a frequency dictionary to a rank dictionary.

    Args:
        freq_dict: Dictionary mapping item → estimated frequency.

    Returns:
        Dictionary mapping item → rank (1 = most frequent).

    Example:
        >>> ranks = compute_rank_dict({0: 500, 1: 300, 2: 200})
        # {0: 1, 1: 2, 2: 3}
    """
    sorted_items = sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True)
    return {item: rank + 1 for rank, (item, _) in enumerate(sorted_items)}
