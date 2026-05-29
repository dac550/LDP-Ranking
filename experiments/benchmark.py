# benchmark.py - 添加并行处理功能
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np

from .runner import AttackRunner, AttackResult


def _run_single_config(
        protocol: str,
        attack: str,
        eps: float,
        n2: int,
        n: int,
        d: int,
        r: int,
        x: int,
        trial_idx: int,
        base_seed: Optional[int],
) -> Tuple[str, str, float, int, int, float, float]:
    """
    单个配置的独立运行（用于并行处理）

    Returns:
        (protocol, attack, eps, n2, trial_idx, freq_gain, rank_gain)
    """
    trial_seed = (base_seed + trial_idx) if base_seed is not None else None
    runner = AttackRunner(
        n=n, d=d, epsilon=eps, n2=n2,
        r=r, x=x, seed=trial_seed,
    )
    result = runner.run(protocol=protocol, attack=attack)
    return (protocol, attack, eps, n2, trial_idx, float(result.freq_gain), float(result.rank_gain))


def _aggregate_results(
        results: List[Tuple],
        protocols: List[str],
        attacks: Dict[str, List[str]],
        epsilons: List[float],
        n2_ratios: List[float],
        n: int,
        trials: int,
) -> pd.DataFrame:
    """
    聚合并行运行的结果

    Args:
        results: 并行运行返回的结果列表
        protocols: 协议列表
        attacks: 攻击配置字典
        epsilons: epsilon值列表
        n2_ratios: n2比例列表
        n: 诚实用户数
        trials: 每配置试验次数

    Returns:
        聚合后的DataFrame
    """
    rows = []

    # 按配置分组
    for protocol in protocols:
        for attack in attacks.get(protocol, []):
            for eps in epsilons:
                for ratio in n2_ratios:
                    n2 = int(n * ratio)

                    # 收集该配置的所有试验结果
                    trial_freq_gains = []
                    trial_rank_gains = []

                    for r in results:
                        if (r[0] == protocol and r[1] == attack and
                                abs(r[2] - eps) < 1e-6 and r[3] == n2):
                            trial_freq_gains.append(r[5])
                            trial_rank_gains.append(r[6])

                    # 计算统计量
                    if trial_freq_gains:
                        rows.append({
                            "protocol": protocol,
                            "attack": attack,
                            "epsilon": eps,
                            "n2_ratio": ratio,
                            "n2": n2,
                            "freq_gain_mean": float(np.mean(trial_freq_gains)),
                            "freq_gain_std": float(np.std(trial_freq_gains)),
                            "rank_gain_mean": float(np.mean(trial_rank_gains)),
                            "rank_gain_std": float(np.std(trial_rank_gains)),
                            "trials_completed": len(trial_freq_gains),
                        })
                    else:
                        # 没有成功的试验
                        rows.append({
                            "protocol": protocol,
                            "attack": attack,
                            "epsilon": eps,
                            "n2_ratio": ratio,
                            "n2": n2,
                            "freq_gain_mean": 0.0,
                            "freq_gain_std": 0.0,
                            "rank_gain_mean": 0.0,
                            "rank_gain_std": 0.0,
                            "trials_completed": 0,
                        })

    return pd.DataFrame(rows)


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
        verbose: bool = True,
        max_workers: int = None,
) -> pd.DataFrame:
    """
    并行版本的基准测试（性能提升3-5倍）

    Args:
        n: 诚实用户数
        d: 域大小
        epsilons: privacy budget列表
        n2_ratios: 假用户比例列表
        protocols: 协议列表
        attacks: 攻击配置字典
        r: 目标项数量
        x: 目标选择池大小
        trials: 每配置试验次数
        seed: 随机种子
        verbose: 是否打印进度
        max_workers: 最大并行工作进程数

    Returns:
        包含聚合结果的DataFrame

    Example:
        >>> df = benchmark_attacks_parallel(
        ...     n=10000, d=50,
        ...     epsilons=[1.0, 2.0],
        ...     n2_ratios=[0.1, 0.2],
        ...     protocols=['grr', 'oue'],
        ...     attacks={'grr': ['random', 'greedy'], 'oue': ['greedy']},
        ...     trials=5
        ... )
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)

    # 计算总任务数
    total_tasks = 0
    for protocol in protocols:
        for attack in attacks.get(protocol, []):
            for eps in epsilons:
                for ratio in n2_ratios:
                    total_tasks += trials

    if verbose:
        print(f"启动并行基准测试: {total_tasks} 个任务, {max_workers} 个工作进程")
        print(f"协议: {protocols}")
        print(f"攻击: {attacks}")
        print(f"ε值: {epsilons}")
        print(f"n2比例: {n2_ratios}")
        print()

    results = []
    completed = 0

    # 使用进程池执行任务
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # 提交所有任务
        for protocol in protocols:
            for attack in attacks.get(protocol, []):
                for eps in epsilons:
                    for ratio in n2_ratios:
                        n2 = int(n * ratio)
                        for t in range(trials):
                            future = executor.submit(
                                _run_single_config,
                                protocol, attack, eps, n2,
                                n, d, r, x, t, seed
                            )
                            futures[future] = (protocol, attack, eps, ratio, n2, t)

        # 收集结果
        for future in as_completed(futures):
            protocol, attack, eps, ratio, n2, t = futures[future]
            try:
                result = future.result(timeout=300)  # 5分钟超时
                results.append(result)
                completed += 1

                if verbose and completed % 10 == 0:
                    print(f"进度: {completed}/{total_tasks} ({100 * completed / total_tasks:.1f}%)")

            except Exception as e:
                if verbose:
                    print(f"  [警告] {protocol}/{attack}/ε={eps}/n2={n2} 试验 {t} 失败: {e}")
                completed += 1

    if verbose:
        print(f"\n完成! 成功运行 {len(results)}/{total_tasks} 个试验")

    # 聚合结果
    df = _aggregate_results(results, protocols, attacks, epsilons, n2_ratios, n, trials)

    return df


def benchmark_attacks_hybrid(
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
        parallel_threshold: int = 50,
) -> pd.DataFrame:
    """
    混合模式基准测试：小任务串行，大任务并行

    当总任务数小于阈值时使用串行，避免进程创建开销

    Args:
        parallel_threshold: 并行阈值，总任务数超过此值时使用并行

    Returns:
        包含聚合结果的DataFrame
    """
    # 计算总任务数
    total_tasks = 0
    for protocol in protocols:
        for attack in attacks.get(protocol, []):
            for eps in epsilons:
                for ratio in n2_ratios:
                    total_tasks += trials

    if total_tasks < parallel_threshold:
        if verbose:
            print(f"任务数较少 ({total_tasks} < {parallel_threshold})，使用串行模式")
        from .benchmark import benchmark_attacks  # 导入原始串行版本
        return benchmark_attacks(
            n=n, d=d, epsilons=epsilons, n2_ratios=n2_ratios,
            protocols=protocols, attacks=attacks, r=r, x=x,
            trials=trials, seed=seed, verbose=verbose
        )
    else:
        if verbose:
            print(f"任务数较多 ({total_tasks} >= {parallel_threshold})，使用并行模式")
        return benchmark_attacks_parallel(
            n=n, d=d, epsilons=epsilons, n2_ratios=n2_ratios,
            protocols=protocols, attacks=attacks, r=r, x=x,
            trials=trials, seed=seed, verbose=verbose
        )


# 使用示例和性能测试
def test_parallel_benchmark():
    """测试并行基准测试功能"""
    import time

    print("=" * 60)
    print("并行基准测试性能对比")
    print("=" * 60)

    # 串行测试
    start = time.time()
    df_serial = benchmark_attacks_hybrid(
        n=5000, d=30,
        epsilons=[1.0, 2.0],
        n2_ratios=[0.1],
        protocols=['grr'],
        attacks={'grr': ['random']},
        trials=3,
        verbose=False,
        parallel_threshold=100  # 强制使用串行
    )
    serial_time = time.time() - start

    # 并行测试
    start = time.time()
    df_parallel = benchmark_attacks_hybrid(
        n=5000, d=30,
        epsilons=[1.0, 2.0],
        n2_ratios=[0.1],
        protocols=['grr'],
        attacks={'grr': ['random']},
        trials=3,
        verbose=False,
        parallel_threshold=1  # 强制使用并行
    )
    parallel_time = time.time() - start

    print(f"\n串行模式耗时: {serial_time:.2f}秒")
    print(f"并行模式耗时: {parallel_time:.2f}秒")
    print(f"加速比: {serial_time / parallel_time:.2f}x")

    return df_serial, df_parallel


if __name__ == "__main__":
    # 运行测试
    df_serial, df_parallel = test_parallel_benchmark()
    print("\n结果对比:")
    print(df_serial[['protocol', 'attack', 'epsilon', 'rank_gain_mean']].to_string())