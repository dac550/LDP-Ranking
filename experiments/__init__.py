"""
LDP Poison Toolkit
==================
A research toolkit for evaluating poisoning attacks on Local Differential
Privacy (LDP) protocols for ranking estimation.

Supported Protocols:
    - GRR  : Generalized Random Response
    - OUE  : Optimized Unary Encoding
    - OLH  : Optimized Local Hashing

Supported Attacks:
    - Random Attack  : Fake users report target items directly.
    - ROA            : Random Output Attack (noise baseline).
    - Greedy Attack  : Iterative cost-minimizing attack.
    - MPOIA          : Greedy with statistical confidence margins (GRR only).

Quick Start:
    >>> from ldp_poison_toolkit.experiments import AttackRunner
    >>> runner = AttackRunner(n=10000, d=50, epsilon=2.0, n2=1000, r=3, x=10)
    >>> result = runner.run(protocol='grr', attack='greedy')
    >>> print(result.summary())

Based on protocols from the Multi-Freq-LDPy package.
"""

from .experiments.runner import AttackRunner, AttackResult
from .experiments.benchmark import benchmark_attacks
from .core import GRR_Client, GRR_Aggregator_MI, UE_Client, UE_Aggregator_MI, LH_Client, LH_Aggregator_MI
from .utils import (
    generate_dataset,
    generate_freq_dict,
    generate_target_items,
    get_gain,
    get_rank_gain,
    grr_estimate,
    oue_estimate,
    olh_estimate,
    expected_perturbed_frequencies,
)
import sys

__version__ = "1.0.0"
__author__ = "LDP Poison Toolkit Contributors"
__license__ = "MIT"

__all__ = [
    # High-level API
    "AttackRunner",
    "AttackResult",
    "benchmark_attacks",
    # Core protocols
    "GRR_Client", "GRR_Aggregator_MI",
    "UE_Client", "UE_Aggregator_MI",
    "LH_Client", "LH_Aggregator_MI",
    # Utilities
    "generate_dataset",
    "generate_freq_dict",
    "generate_target_items",
    "get_gain",
    "get_rank_gain",
    "grr_estimate",
    "oue_estimate",
    "olh_estimate",
    "expected_perturbed_frequencies",
]

def _check_python_version():
    """检查 Python 版本兼容性"""
    if sys.version_info < (3, 8):
        raise RuntimeError(
            f"LDP Poison Toolkit 需要 Python 3.8 或更高版本，"
            f"当前版本: {sys.version_info.major}.{sys.version_info.minor}"
        )


def _check_dependencies():
    """检查依赖包版本"""
    import importlib.metadata as metadata

    required = {
        'numpy': '1.21',
        'scipy': '1.7',
        'pandas': '1.3',
    }

    for pkg, min_version in required.items():
        try:
            version = metadata.version(pkg)
            if version < min_version:
                print(f"警告: {pkg} 版本 {version} < {min_version}，可能存在兼容性问题")
        except metadata.PackageNotFoundError:
            print(f"错误: 未找到 {pkg} 包")
            raise


# 初始化时运行检查
_check_python_version()
_check_dependencies()
