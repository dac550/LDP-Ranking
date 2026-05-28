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


def plot_benchmark_results(df, metric='rank_gain_mean', save_path=None):
    """
    绘制基准测试结果

    Args:
        df: benchmark_attacks 返回的 DataFrame
        metric: 要绘制的指标
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, protocol in enumerate(df['protocol'].unique()):
        ax = axes[idx]
        protocol_df = df[df['protocol'] == protocol]

        for attack in protocol_df['attack'].unique():
            attack_df = protocol_df[protocol_df['attack'] == attack]
            for eps in attack_df['epsilon'].unique():
                eps_df = attack_df[attack_df['epsilon'] == eps]
                ax.plot(eps_df['n2_ratio'], eps_df[metric],
                        marker='o', label=f'{attack}, ε={eps}')

        ax.set_xlabel('Fake User Ratio (n2/n)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{protocol.upper()} Protocol')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


def _run_single_trial(
        protocol: str,
        attack: str,
        eps: float,
        n2: int,
        n: int,
        d: int,
        r: int,
        x: int,
        seed: int,
) -> tuple:
    """单个试验的运行函数（用于并行）"""
    try:
        runner = AttackRunner.from_params(
            n=n, d=d, epsilon=eps, n2=n2,
            r=r, x=x, seed=seed
        )
        result = runner.run(protocol=protocol, attack=attack)
        return result.freq_gain, result.rank_gain, None
    except Exception as e:
        return 0, 0, str(e)


def benchmark_attacks_parallel(
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
        max_workers: int = 4,
        verbose: bool = True,
) -> pd.DataFrame:
    """并行版本的基准测试"""
    rows = []

    for protocol in protocols:
        for attack in attacks.get(protocol, []):
            for eps in epsilons:
                for ratio in n2_ratios:
                    n2 = int(n * ratio)

                    # 准备所有试验的种子
                    base_seed = seed if seed is not None else 0
                    seeds = [base_seed + i for i in range(trials)]

                    # 并行执行
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        fn = partial(
                            _run_single_trial,
                            protocol=protocol,
                            attack=attack,
                            eps=eps,
                            n2=n2,
                            n=n,
                            d=d,
                            r=r,
                            x=x,
                        )
                        futures = [executor.submit(fn, seed=s) for s in seeds]

                        freq_gains = []
                        rank_gains = []
                        errors = []

                        for future in as_completed(futures):
                            fg, rg, err = future.result()
                            if err:
                                errors.append(err)
                            else:
                                freq_gains.append(fg)
                                rank_gains.append(rg)

                    if verbose:
                        print(f"[{protocol.upper()}] {attack} | ε={eps} | "
                              f"n2={n2} | rank_gain={np.mean(rank_gains):.1f}±{np.std(rank_gains):.1f}")

                    rows.append({
                        "protocol": protocol,
                        "attack": attack,
                        "epsilon": eps,
                        "n2_ratio": ratio,
                        "n2": n2,
                        "freq_gain_mean": np.mean(freq_gains) if freq_gains else 0,
                        "freq_gain_std": np.std(freq_gains) if freq_gains else 0,
                        "rank_gain_mean": np.mean(rank_gains) if rank_gains else 0,
                        "rank_gain_std": np.std(rank_gains) if rank_gains else 0,
                        "errors": len(errors),
                    })

    return pd.DataFrame(rows)