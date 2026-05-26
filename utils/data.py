"""
Data Generation Utilities
==========================
Synthetic dataset generation and preprocessing helpers for LDP experiments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def generate_dataset(n: int, d: int, m: float = 10.0) -> pd.Series:
    """
    Generate a synthetic dataset following a Zipf-like frequency distribution.

    The dataset is constructed as follows:
      1. Draw d random values from Uniform[1, m].
      2. Normalize them to sum to n (floor), giving integer counts.
      3. Sort counts in descending order.
      4. Assign the i-th largest count to item i.

    Args:
        n: Total number of data points.
        d: Number of distinct items (domain size).
        m: Upper bound for uniform sampling (controls skewness). Default 10.

    Returns:
        A pandas Series of length n with integer values in [0, d-1].

    Example:
        >>> dataset = generate_dataset(n=10000, d=50, m=10.0)
    """
    c = np.random.uniform(1, m, d)
    c_prime = np.floor((c / c.sum()) * n).astype(int)
    c_sorted = np.sort(c_prime)[::-1]

    values = np.arange(d)
    repeated = np.repeat(values, c_sorted)
    return pd.Series(repeated)


def generate_freq_dict(dataset: pd.Series) -> Dict[int, int]:
    """
    Compute frequency counts for each item in the dataset.

    Args:
        dataset: A pandas Series of integer item values.

    Returns:
        Dictionary mapping item → count, sorted by count descending.

    Example:
        >>> freq = generate_freq_dict(dataset)
    """
    value_counts = dataset.value_counts().sort_index()
    freq = value_counts.to_dict()
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))


def generate_target_items(freq_dict: Dict[int, int], r: int, x: int) -> np.ndarray:
    """
    Select r target items uniformly at random from the top-x most frequent items.

    Target items represent the items an adversary wishes to promote in the
    ranking. Selecting from the top-x ensures they are realistic candidates
    (already somewhat popular, but potentially improvable).

    Args:
        freq_dict: Frequency dictionary (item → count), sorted descending.
        r: Number of target items to select.
        x: Pool size — sample from the top-x items only.

    Returns:
        Sorted numpy array of r selected target items.

    Raises:
        ValueError: If r > x or x > len(freq_dict).

    Example:
        >>> targets = generate_target_items(freq_dict, r=3, x=10)
    """
    if r > x:
        raise ValueError(f"r ({r}) cannot exceed x ({x}).")
    if x > len(freq_dict):
        raise ValueError(f"x ({x}) exceeds number of items ({len(freq_dict)}).")

    sorted_items = sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True)[:x]
    top_x = [item for item, _ in sorted_items]

    targets = np.random.choice(top_x, r, replace=False)
    return np.sort(targets)
