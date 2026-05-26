"""
Test Suite
===========
Unit and integration tests for the LDP Poison Toolkit.

Run:
    pytest tests/test_toolkit.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ldp_poison_toolkit.core.grr import GRR_Client, GRR_Aggregator_MI
from ldp_poison_toolkit.core.oue import UE_Client, UE_Aggregator_MI
from ldp_poison_toolkit.core.olh import LH_Client, LH_Aggregator_MI
from ldp_poison_toolkit.utils.data import generate_dataset, generate_freq_dict, generate_target_items
from ldp_poison_toolkit.utils.metrics import get_gain, get_rank_gain, compute_rank_dict
from ldp_poison_toolkit.utils.estimators import (
    grr_estimate, oue_estimate, olh_estimate, expected_perturbed_frequencies
)
from ldp_poison_toolkit.attacks.grr_attacks import (
    random_attack_grr, greedy_attack_grr, mpoia_attack_grr
)
from ldp_poison_toolkit.attacks.oue_attacks import (
    random_attack_oue, roa_attack_oue, greedy_attack_oue
)
from ldp_poison_toolkit.attacks.olh_attacks import (
    random_attack_olh, roa_attack_olh
)
from ldp_poison_toolkit.experiments.runner import AttackRunner


# ─── Core Protocol Tests ─────────────────────────────────────────────────────

class TestGRR:
    def test_client_returns_valid_value(self):
        d, eps = 10, 2.0
        for v in range(d):
            out = GRR_Client(v, d, eps)
            assert 0 <= out < d

    def test_client_invalid_input(self):
        with pytest.raises(ValueError):
            GRR_Client(-1, 10, 2.0)
        with pytest.raises(ValueError):
            GRR_Client(10, 10, 2.0)
        with pytest.raises(ValueError):
            GRR_Client(0, 10, -1.0)

    def test_aggregator_output_shape(self):
        d, eps, n = 10, 2.0, 500
        reports = [GRR_Client(np.random.randint(0, d), d, eps) for _ in range(n)]
        est = GRR_Aggregator_MI(reports, d, eps)
        assert len(est) == d
        assert all(v >= 0 for v in est)

    def test_aggregator_empty_raises(self):
        with pytest.raises(ValueError):
            GRR_Aggregator_MI([], 10, 2.0)


class TestOUE:
    def test_client_returns_binary_vector(self):
        d, eps = 20, 2.0
        vec = UE_Client(5, d, eps)
        assert len(vec) == d
        assert set(vec).issubset({0, 1})

    def test_aggregator_shape(self):
        d, eps, n = 20, 2.0, 300
        reports = [UE_Client(np.random.randint(0, d), d, eps) for _ in range(n)]
        est = UE_Aggregator_MI(reports, d, eps)
        assert len(est) == d

    def test_aggregator_empty_raises(self):
        with pytest.raises(ValueError):
            UE_Aggregator_MI([], 20, 2.0)


class TestOLH:
    def test_client_returns_tuple(self):
        d, eps = 50, 2.0
        result = LH_Client(7, d, eps)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_aggregator_shape(self):
        d, eps, n = 20, 2.0, 100
        reports = [LH_Client(np.random.randint(0, d), d, eps) for _ in range(n)]
        est = LH_Aggregator_MI(reports, d, eps)
        assert len(est) == d


# ─── Data Utility Tests ───────────────────────────────────────────────────────

class TestDataUtils:
    def test_dataset_length(self):
        ds = generate_dataset(n=1000, d=20)
        assert len(ds) <= 1000  # floor may reduce by a few

    def test_freq_dict_keys(self):
        ds = generate_dataset(n=1000, d=20)
        fd = generate_freq_dict(ds)
        assert all(k in range(20) for k in fd)

    def test_target_items_count(self):
        ds = generate_dataset(n=1000, d=20)
        fd = generate_freq_dict(ds)
        targets = generate_target_items(fd, r=3, x=8)
        assert len(targets) == 3

    def test_target_items_sorted(self):
        ds = generate_dataset(n=1000, d=20)
        fd = generate_freq_dict(ds)
        targets = generate_target_items(fd, r=3, x=8)
        assert list(targets) == sorted(targets)


# ─── Metric Tests ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_freq_gain_positive(self):
        before = {0: 100, 1: 200}
        after  = {0: 150, 1: 200}
        assert get_gain([0], before, after) == 50

    def test_rank_gain(self):
        before_ranks = {0: 3, 1: 1, 2: 2}
        after_ranks  = {0: 1, 1: 2, 2: 3}
        assert get_rank_gain([0], before_ranks, after_ranks) == 2  # 3 -> 1

    def test_compute_rank_dict(self):
        ranks = compute_rank_dict({0: 500, 1: 300, 2: 100})
        assert ranks[0] == 1
        assert ranks[1] == 2
        assert ranks[2] == 3


# ─── Attack Tests ─────────────────────────────────────────────────────────────

class TestGRRAttacks:
    def setup_method(self):
        np.random.seed(0)
        self.d, self.eps, self.n = 20, 2.0, 1000
        self.T = [0, 1, 2]
        self.A = list(range(3, 20))
        ds = generate_dataset(self.n, self.d)
        fd = generate_freq_dict(ds)
        p = np.exp(self.eps) / (np.exp(self.eps) + self.d - 1)
        q = (1 - p) / (self.d - 1)
        self.exp_freq = expected_perturbed_frequencies(fd, self.n, p, q)
        self.ranks = compute_rank_dict(self.exp_freq)

    def test_random_attack_length(self):
        fake = random_attack_grr(100, self.T)
        assert len(fake) == 100
        assert all(v in self.T for v in fake)

    def test_greedy_attack_length(self):
        fake = greedy_attack_grr(100, self.T, self.A, self.exp_freq, self.ranks)
        assert len(fake) == 100

    def test_mpoia_attack_length(self):
        ds = generate_dataset(self.n, self.d)
        fd = generate_freq_dict(ds)
        p = np.exp(self.eps) / (np.exp(self.eps) + self.d - 1)
        fake = mpoia_attack_grr(100, self.T, self.A, fd, self.exp_freq, self.ranks, self.n, p)
        assert len(fake) == 100


class TestOUEAttacks:
    def setup_method(self):
        np.random.seed(0)
        self.d, self.eps = 20, 2.0
        self.T = [0, 1]
        self.A = list(range(2, 20))

    def test_random_attack(self):
        fake = random_attack_oue(50, self.T, self.d)
        assert len(fake) == 50
        assert all(len(v) == self.d for v in fake)

    def test_roa_attack(self):
        fake = roa_attack_oue(50, self.A, self.d, self.eps)
        assert len(fake) == 50

    def test_greedy_attack(self):
        ds = generate_dataset(1000, self.d)
        fd = generate_freq_dict(ds)
        p, q = 0.5, 1.0 / (np.exp(self.eps) + 1)
        exp_freq = expected_perturbed_frequencies(fd, 1000, p, q)
        ranks = compute_rank_dict(exp_freq)
        fake = greedy_attack_oue(50, self.T, self.A, exp_freq, ranks, self.d, self.eps)
        assert len(fake) == 50


# ─── Integration Tests ────────────────────────────────────────────────────────

class TestAttackRunner:
    def test_grr_random(self):
        runner = AttackRunner(n=2000, d=20, epsilon=2.0, n2=200, r=2, x=6, seed=42)
        result = runner.run(protocol='grr', attack='random')
        assert result.protocol == 'grr'
        assert result.attack == 'random'
        assert isinstance(result.freq_gain, (int, float))

    def test_grr_greedy(self):
        runner = AttackRunner(n=2000, d=20, epsilon=2.0, n2=200, r=2, x=6, seed=0)
        result = runner.run(protocol='grr', attack='greedy')
        assert isinstance(result.rank_gain, (int, float))

    def test_oue_greedy(self):
        runner = AttackRunner(n=2000, d=20, epsilon=2.0, n2=200, r=2, x=6, seed=1)
        result = runner.run(protocol='oue', attack='greedy')
        assert len(result.after_freq) == 20

    def test_invalid_protocol(self):
        runner = AttackRunner(n=1000, d=10, epsilon=1.0, n2=100, r=2, x=5)
        with pytest.raises(ValueError):
            runner.run(protocol='xyz', attack='random')

    def test_summary_str(self):
        runner = AttackRunner(n=1000, d=10, epsilon=1.0, n2=100, r=2, x=5, seed=7)
        result = runner.run(protocol='grr', attack='random')
        s = result.summary()
        assert "GRR" in s
        assert "random" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
