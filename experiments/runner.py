"""
Attack Runner
==============
High-level experiment runner for evaluating poisoning attacks on LDP protocols.
Handles the full pipeline: dataset generation → LDP estimation → attack injection
→ re-estimation → metric computation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..utils.data import generate_dataset, generate_freq_dict, generate_target_items
from ..utils.metrics import get_gain, get_rank_gain, compute_rank_dict
from ..utils.estimators import (
    grr_estimate,
    oue_estimate,
    olh_estimate,
    expected_perturbed_frequencies,
)
from ..attacks.grr_attacks import random_attack_grr, greedy_attack_grr, mpoia_attack_grr
from ..attacks.oue_attacks import random_attack_oue, roa_attack_oue, greedy_attack_oue
from ..attacks.olh_attacks import random_attack_olh, roa_attack_olh, greedy_attack_olh


@dataclass
class AttackResult:
    """
    Container for a single attack experiment result.

    Attributes:
        protocol: LDP protocol used ('grr', 'oue', 'olh').
        attack: Attack strategy name.
        epsilon: Privacy budget.
        n: Honest user count.
        n2: Fake user count.
        d: Domain size.
        r: Number of target items.
        target_items: Selected target item IDs.
        freq_gain: Total frequency gain for target items.
        rank_gain: Total rank improvement for target items.
        before_freq: Estimated frequencies before attack.
        after_freq: Estimated frequencies after attack.
        before_ranks: Ranks before attack.
        after_ranks: Ranks after attack.
    """
    protocol: str
    attack: str
    epsilon: float
    n: int
    n2: int
    d: int
    r: int
    target_items: List[int]
    freq_gain: int = 0
    rank_gain: int = 0
    before_freq: Dict[int, int] = field(default_factory=dict)
    after_freq: Dict[int, int] = field(default_factory=dict)
    before_ranks: Dict[int, int] = field(default_factory=dict)
    after_ranks: Dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Protocol   : {self.protocol.upper()}",
            f"Attack     : {self.attack}",
            f"ε          : {self.epsilon}",
            f"n / n2     : {self.n} / {self.n2} ({100*self.n2/(self.n+self.n2):.1f}% fake)",
            f"d / r      : {self.d} / {self.r}",
            f"Targets    : {list(self.target_items)}",
            f"Freq gain  : {self.freq_gain:+d}",
            f"Rank gain  : {self.rank_gain:+d}",
        ]
        for t in self.target_items:
            br = self.before_ranks.get(t, "?")
            ar = self.after_ranks.get(t, "?")
            lines.append(f"  item {t:3d} : rank {br} → {ar}")
        return "\n".join(lines)


import logging
from datetime import datetime
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttackRunner:
    """
    Orchestrates end-to-end poisoning attack experiments on LDP protocols.

    Usage:
        runner = AttackRunner(n=10000, d=50, epsilon=2.0, n2=1000, r=3, x=10)
        result = runner.run(protocol='grr', attack='greedy')
        print(result.summary())
    """

    def __init__(
        self,
        n: int,
        d: int,
        epsilon: float,
        n2: int,
        r: int = 3,
        x: int = 10,
        m: float = 10.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the experiment runner.

        Args:
            n: Number of honest users.
            d: Domain size (number of distinct items).
            epsilon: LDP privacy budget.
            n2: Number of fake (adversarial) users to inject.
            r: Number of target items.
            x: Pool size for target item selection (from top-x items).
            m: Skewness parameter for dataset generation.
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        self.n = n
        self.d = d
        self.epsilon = epsilon
        self.n2 = n2
        self.r = r
        self.x = x

        # Generate dataset
        self.dataset = generate_dataset(n, d, m)
        self.true_freq = generate_freq_dict(self.dataset)
        self.target_items = generate_target_items(self.true_freq, r, x)
        self.non_target_items = [i for i in range(d) if i not in self.target_items]


    def run(self, protocol: str, attack: str, **kwargs) -> AttackResult:
        """
        Run a single attack experiment.

        Args:
            protocol: One of 'grr', 'oue', 'olh'.
            attack: One of 'random', 'roa', 'greedy', 'mpoia'.
            **kwargs: Additional keyword arguments forwarded to the attack function.

        Returns:
            AttackResult with all metrics populated.

        Raises:
            ValueError: For unknown protocol or attack name.
        """
        logger.info(f"[{datetime.now()}] 开始实验: protocol={protocol}, attack={attack}")
        logger.info(f"参数: n={self.n}, n2={self.n2}, d={self.d}, ε={self.epsilon}")
        protocol = protocol.lower()
        attack = attack.lower()

        # --- Step 1: Honest estimation ---
        if protocol == 'grr':
            per_data, before_freq, before_ranks = grr_estimate(self.dataset, self.d, self.epsilon)
            p = np.exp(self.epsilon) / (np.exp(self.epsilon) + self.d - 1)
            q = (1 - p) / (self.d - 1)
        elif protocol == 'oue':
            per_data, before_freq, before_ranks = oue_estimate(self.dataset, self.d, self.epsilon)
            p = 0.5
            q = 1.0 / (np.exp(self.epsilon) + 1)
        elif protocol == 'olh':
            per_data, before_freq, before_ranks = olh_estimate(self.dataset, self.d, self.epsilon)
            g = int(round(np.exp(self.epsilon))) + 1
            p = np.exp(self.epsilon) / (np.exp(self.epsilon) + g - 1)
            q = 1.0 / g
        else:
            raise ValueError(f"Unknown protocol '{protocol}'. Choose from: grr, oue, olh.")

        # --- Step 2: Compute expected perturbed frequencies ---
        exp_freq = expected_perturbed_frequencies(self.true_freq, self.n, p, q)

        # --- Step 3: Generate fake data ---
        T = list(self.target_items)
        A = self.non_target_items

        if protocol == 'grr':
            fake = self._grr_fake(attack, T, A, exp_freq, before_ranks, **kwargs)
        elif protocol == 'oue':
            fake = self._oue_fake(attack, T, A, exp_freq, before_ranks, **kwargs)
        elif protocol == 'olh':
            fake = self._olh_fake(attack, T, A, exp_freq, before_ranks, **kwargs)

        # --- Step 4: Combine and re-estimate ---
        combined = list(per_data) + list(fake)

        if protocol == 'grr':
            _, after_freq, after_ranks = grr_estimate(combined, self.d, self.epsilon, perturb=False)
        elif protocol == 'oue':
            _, after_freq, after_ranks = oue_estimate(combined, self.d, self.epsilon, perturb=False)
        elif protocol == 'olh':
            _, after_freq, after_ranks = olh_estimate(combined, self.d, self.epsilon, perturb=False)

        # --- Step 5: Compute metrics ---
        freq_gain = get_gain(T, before_freq, after_freq)
        rank_gain = get_rank_gain(T, before_ranks, after_ranks)
        logger.info(f"完成! 频数增益={freq_gain}, 排名增益={rank_gain}")
        return AttackResult(
            protocol=protocol,
            attack=attack,
            epsilon=self.epsilon,
            n=self.n,
            n2=self.n2,
            d=self.d,
            r=self.r,
            target_items=T,
            freq_gain=freq_gain,
            rank_gain=rank_gain,
            before_freq=before_freq,
            after_freq=after_freq,
            before_ranks=before_ranks,
            after_ranks=after_ranks,
        )

    # -----------------------------------------------------------------------
    # Private dispatch helpers
    # -----------------------------------------------------------------------

    def _grr_fake(self, attack, T, A, exp_freq, before_ranks, **kwargs):
        if attack == 'random':
            return random_attack_grr(self.n2, T)
        elif attack == 'greedy':
            return greedy_attack_grr(self.n2, T, A, exp_freq, before_ranks)
        elif attack == 'mpoia':
            p = np.exp(self.epsilon) / (np.exp(self.epsilon) + self.d - 1)
            return mpoia_attack_grr(
                self.n2, T, A, self.true_freq, exp_freq, before_ranks,
                self.n, p, kwargs.get('confidence', 0.9)
            )
        raise ValueError(f"Unknown GRR attack '{attack}'. Choose from: random, greedy, mpoia.")

    def _oue_fake(self, attack, T, A, exp_freq, before_ranks, **kwargs):
        if attack == 'random':
            return random_attack_oue(self.n2, T, self.d)
        elif attack == 'roa':
            return roa_attack_oue(self.n2, A, self.d, self.epsilon)
        elif attack == 'greedy':
            return greedy_attack_oue(self.n2, T, A, exp_freq, before_ranks, self.d, self.epsilon)
        raise ValueError(f"Unknown OUE attack '{attack}'. Choose from: random, roa, greedy.")

    def _olh_fake(self, attack, T, A, exp_freq, before_ranks, **kwargs):
        if attack == 'random':
            return random_attack_olh(self.n2, T, self.d, self.epsilon)
        elif attack == 'roa':
            return roa_attack_olh(self.n2, A, self.d, self.epsilon)
        elif attack == 'greedy':
            return greedy_attack_olh(
                self.n2, T, A, exp_freq, self.d, self.epsilon,
                kwargs.get('hash_funcs'), kwargs.get('num_hash_funcs', 100)
            )
        raise ValueError(f"Unknown OLH attack '{attack}'. Choose from: random, roa, greedy.")


from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExperimentConfig:
    """实验配置类（不可变）"""
    n: int  # 诚实用户数
    d: int  # 域大小
    epsilon: float  # 隐私预算
    n2: int  # 假用户数
    r: int = 3  # 目标项数量
    x: int = 10  # 目标选择池大小
    m: float = 10.0  # 数据倾斜参数
    seed: Optional[int] = None

    def __post_init__(self):
        """参数验证"""
        if self.n <= 0:
            raise ValueError(f"n must be > 0, got {self.n}")
        if self.d < 2:
            raise ValueError(f"d must be >= 2, got {self.d}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.n2 < 0:
            raise ValueError(f"n2 must be >= 0, got {self.n2}")
        if self.r > self.d:
            raise ValueError(f"r ({self.r}) cannot exceed d ({self.d})")
        if self.x > self.d:
            raise ValueError(f"x ({self.x}) cannot exceed d ({self.d})")
        if self.r > self.x:
            raise ValueError(f"r ({self.r}) cannot exceed x ({self.x})")

class AttackRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)

        self.dataset = generate_dataset(config.n, config.d, config.m)
        self.true_freq = generate_freq_dict(self.dataset)
        self.target_items = generate_target_items(self.true_freq, config.r, config.x)
        self.non_target_items = [i for i in range(config.d) if i not in self.target_items]

    # 兼容旧接口的构造函数
    @classmethod
    def from_params(cls, n: int, d: int, epsilon: float, n2: int,
                    r: int = 3, x: int = 10, m: float = 10.0, seed: int = None):
        config = ExperimentConfig(n=n, d=d, epsilon=epsilon, n2=n2,
                                  r=r, x=x, m=m, seed=seed)
        return cls(config)