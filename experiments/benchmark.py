"""
Benchmark Utilities
====================
Run multiple attack configurations across protocols and aggregate results.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from .runner import AttackRunner, AttackResult


def benchmark_attacks(
    n: int,
    d: int,
    epsilons: List[float],
    n2_ratios: List[float],
    protocols: List[str],
    attacks: Dict[str, List[str]],
    r: int = 3,
    x: int = 10,
    trials: int = 5,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Benchmark multiple attack configurations across protocols and epsilons.

    Runs each (protocol, attack, epsilon, n2_ratio) combination for `trials`
    independent trials and aggregates mean and std of freq_gain and rank_gain.

    Args:
        n: Honest user count.
        d: Domain size.
        epsilons: List of privacy budgets to test.
        n2_ratios: List of ratios n2/n (e.g. [0.1, 0.2, 0.3]).
        protocols: List of protocols to test (e.g. ['grr', 'oue']).
        attacks: Dict mapping protocol → list of attack names.
                 e.g. {'grr': ['random', 'greedy'], 'oue': ['greedy']}
        r: Number of target items.
        x: Pool size for target selection.
        trials: Number of independent trials per configuration.
        seed: Base random seed.
        verbose: Print progress.

    Returns:
        DataFrame with columns:
            protocol, attack, epsilon, n2_ratio, n2,
            freq_gain_mean, freq_gain_std,
            rank_gain_mean, rank_gain_std.

    Example:
        >>> df = benchmark_attacks(
        ...     n=10000, d=50,
        ...     epsilons=[1.0, 2.0, 4.0],
        ...     n2_ratios=[0.1, 0.2],
        ...     protocols=['grr', 'oue'],
        ...     attacks={'grr': ['random', 'greedy'], 'oue': ['greedy']},
        ... )
        >>> print(df.to_string())
    """
    rows = []
    total = len(protocols) * len(epsilons) * len(n2_ratios) * trials
    done = 0

    for protocol in protocols:
        attack_list = attacks.get(protocol, [])
        for attack in attack_list:
            for eps in epsilons:
                for ratio in n2_ratios:
                    n2 = int(n * ratio)
                    trial_freq_gains, trial_rank_gains = [], []

                    for t in range(trials):
                        trial_seed = (seed + done) if seed is not None else None
                        try:
                            runner = AttackRunner(
                                n=n, d=d, epsilon=eps, n2=n2,
                                r=r, x=x, seed=trial_seed,
                            )
                            result = runner.run(protocol=protocol, attack=attack)
                            trial_freq_gains.append(result.freq_gain)
                            trial_rank_gains.append(result.rank_gain)
                        except Exception as e:
                            if verbose:
                                print(f"  [WARN] {protocol}/{attack}/ε={eps}/n2={n2} trial {t}: {e}")
                        done += 1

                    if verbose:
                        print(
                            f"[{protocol.upper():3s}] {attack:8s} | ε={eps:.1f} | "
                            f"n2={n2:5d} ({100*ratio:.0f}%) | "
                            f"rank_gain={np.mean(trial_rank_gains):.1f}±{np.std(trial_rank_gains):.1f}"
                        )

                    rows.append({
                        "protocol": protocol,
                        "attack": attack,
                        "epsilon": eps,
                        "n2_ratio": ratio,
                        "n2": n2,
                        "freq_gain_mean": float(np.mean(trial_freq_gains)) if trial_freq_gains else 0,
                        "freq_gain_std": float(np.std(trial_freq_gains)) if trial_freq_gains else 0,
                        "rank_gain_mean": float(np.mean(trial_rank_gains)) if trial_rank_gains else 0,
                        "rank_gain_std": float(np.std(trial_rank_gains)) if trial_rank_gains else 0,
                    })

    return pd.DataFrame(rows)
